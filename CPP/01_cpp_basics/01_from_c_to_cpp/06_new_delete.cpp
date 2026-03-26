/**
 * @file 06_new_delete.cpp
 * @brief C++动态内存管理 - new/delete vs malloc/free
 * 
 * 【推理加速场景】
 * 推理引擎需要精细控制内存：
 * - 大块连续内存分配
 * - 对齐内存（SIMD/GPU需要）
 * - 内存池减少分配开销
 * 
 * 编译: g++ -std=c++17 -o 06_new_delete 06_new_delete.cpp
 */

#include <iostream>
#include <new>       // placement new
#include <cstdlib>   // aligned_alloc

// ============================================================================
// 第一部分：new/delete 基础
// ============================================================================

class Tensor {
public:
    float* data_;
    int size_;
    
    Tensor(int size) : size_(size) {
        data_ = new float[size];  // 分配数组
        std::cout << "Tensor构造: 分配 " << size * sizeof(float) << " 字节\n";
    }
    
    ~Tensor() {
        delete[] data_;  // 释放数组
        std::cout << "Tensor析构: 释放内存\n";
    }
};

void demo_new_delete() {
    std::cout << "\n=== new/delete 基础 ===\n";
    
    // 单个对象
    int* p = new int(42);
    std::cout << "*p = " << *p << "\n";
    delete p;
    
    // 数组
    float* arr = new float[1024];
    arr[0] = 1.0f;
    delete[] arr;  // 注意：数组用delete[]
    
    // 自动调用构造/析构
    Tensor* t = new Tensor(1024);
    delete t;
}

// ============================================================================
// 第二部分：对齐内存分配
// ============================================================================

void demo_aligned_alloc() {
    std::cout << "\n=== 对齐内存 (SIMD必需) ===\n";
    
    // AVX2需要32字节对齐，AVX-512需要64字节对齐
    constexpr size_t ALIGNMENT = 64;
    constexpr size_t SIZE = 1024 * sizeof(float);
    
    // C++17 aligned_alloc
    void* aligned_ptr = std::aligned_alloc(ALIGNMENT, SIZE);
    if (aligned_ptr) {
        std::cout << "分配 " << SIZE << " 字节，对齐到 " << ALIGNMENT << " 字节\n";
        std::cout << "地址: " << aligned_ptr 
                  << " (模 " << ALIGNMENT << " = " 
                  << (reinterpret_cast<uintptr_t>(aligned_ptr) % ALIGNMENT) << ")\n";
        std::free(aligned_ptr);
    }
    
    // C++17 operator new with alignment
    float* aligned_arr = static_cast<float*>(
        ::operator new(SIZE, std::align_val_t{ALIGNMENT}));
    ::operator delete(aligned_arr, std::align_val_t{ALIGNMENT});
}

// ============================================================================
// 第三部分：Placement New - 在指定位置构造对象
// ============================================================================

void demo_placement_new() {
    std::cout << "\n=== Placement New (内存池常用) ===\n";
    
    // 预分配缓冲区
    alignas(Tensor) char buffer[sizeof(Tensor)];
    
    // 在缓冲区中构造对象
    Tensor* t = new (buffer) Tensor(512);
    
    // 手动调用析构函数（不能delete，因为内存不是new分配的）
    t->~Tensor();
    
    std::cout << "对象在栈上缓冲区中构造和析构\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "C++ 动态内存管理\n";
    std::cout << "========================================\n";
    
    demo_new_delete();
    demo_aligned_alloc();
    demo_placement_new();
    
    std::cout << "\n总结: 推理优化常用对齐分配和内存池技术\n";
    std::cout << "后续将学习智能指针，更安全地管理内存\n";
    return 0;
}
