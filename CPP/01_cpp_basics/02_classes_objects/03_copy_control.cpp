/**
 * @file 03_copy_control.cpp
 * @brief 拷贝控制 - 深拷贝、浅拷贝与Rule of Five
 * 
 * 【推理加速场景】
 * Tensor包含大量数据，必须正确处理拷贝：
 * - 意外的深拷贝会导致性能问题
 * - 浅拷贝可能导致双重释放
 * - 移动语义实现零拷贝传递
 * 
 * 编译: g++ -std=c++17 -o 03_copy_control 03_copy_control.cpp
 */

#include <iostream>
#include <vector>
#include <algorithm>

// ============================================================================
// 第一部分：浅拷贝的问题
// ============================================================================

class DangerousTensor {
public:
    float* data;
    size_t size;
    
    DangerousTensor(size_t n) : size(n) {
        data = new float[n]();
        std::cout << "[创建] DangerousTensor at " << data << "\n";
    }
    
    // 没有定义拷贝构造函数，编译器生成默认的浅拷贝
    // 危险！两个对象指向同一块内存
    
    ~DangerousTensor() {
        std::cout << "[销毁] DangerousTensor at " << data << "\n";
        delete[] data;  // 双重释放！
    }
};

// ============================================================================
// 第二部分：Rule of Five（五法则）
// ============================================================================

/**
 * 如果定义了以下任一函数，通常应该定义全部五个：
 * 1. 析构函数
 * 2. 拷贝构造函数
 * 3. 拷贝赋值运算符
 * 4. 移动构造函数
 * 5. 移动赋值运算符
 */

class SafeTensor {
private:
    float* data_;
    size_t size_;
    std::string name_;
    
public:
    // 构造函数
    SafeTensor(const std::string& name, size_t size) 
        : name_(name), size_(size) {
        data_ = new float[size_]();
        std::cout << "[构造] " << name_ << " at " << data_ << "\n";
    }
    
    // 1. 析构函数
    ~SafeTensor() {
        std::cout << "[析构] " << name_ << " at " << data_ << "\n";
        delete[] data_;
    }
    
    // 2. 拷贝构造函数（深拷贝）
    SafeTensor(const SafeTensor& other) 
        : name_(other.name_ + "_copy"), size_(other.size_) {
        data_ = new float[size_];
        std::copy(other.data_, other.data_ + size_, data_);
        std::cout << "[拷贝构造] " << name_ << " from " << other.name_ << "\n";
    }
    
    // 3. 拷贝赋值运算符
    SafeTensor& operator=(const SafeTensor& other) {
        std::cout << "[拷贝赋值] " << name_ << " = " << other.name_ << "\n";
        if (this != &other) {  // 自赋值检查
            // 先释放旧资源
            delete[] data_;
            // 分配新资源
            size_ = other.size_;
            data_ = new float[size_];
            std::copy(other.data_, other.data_ + size_, data_);
            name_ = other.name_ + "_assigned";
        }
        return *this;
    }
    
    // 4. 移动构造函数
    SafeTensor(SafeTensor&& other) noexcept 
        : data_(other.data_), size_(other.size_), 
          name_(std::move(other.name_)) {
        std::cout << "[移动构造] " << name_ << "\n";
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    // 5. 移动赋值运算符
    SafeTensor& operator=(SafeTensor&& other) noexcept {
        std::cout << "[移动赋值] " << name_ << " <- " << other.name_ << "\n";
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            name_ = std::move(other.name_);
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // 辅助函数
    const std::string& name() const { return name_; }
    size_t size() const { return size_; }
};

// ============================================================================
// 第三部分：禁止拷贝
// ============================================================================

class NonCopyableTensor {
private:
    float* data_;
    size_t size_;
    
public:
    NonCopyableTensor(size_t size) : size_(size) {
        data_ = new float[size_]();
    }
    
    ~NonCopyableTensor() { delete[] data_; }
    
    // 禁止拷贝
    NonCopyableTensor(const NonCopyableTensor&) = delete;
    NonCopyableTensor& operator=(const NonCopyableTensor&) = delete;
    
    // 允许移动
    NonCopyableTensor(NonCopyableTensor&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
    }
    
    NonCopyableTensor& operator=(NonCopyableTensor&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
        }
        return *this;
    }
};

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "拷贝控制详解\n";
    std::cout << "========================================\n";
    
    // 注意：以下代码会崩溃，仅作演示
    // std::cout << "\n--- 1. 浅拷贝的危险 ---\n";
    // {
    //     DangerousTensor t1(100);
    //     DangerousTensor t2 = t1;  // 浅拷贝
    // }  // 双重释放！程序崩溃
    
    std::cout << "\n--- 2. 安全的拷贝 ---\n";
    {
        SafeTensor t1("original", 100);
        SafeTensor t2 = t1;  // 拷贝构造（深拷贝）
        SafeTensor t3("other", 50);
        t3 = t1;  // 拷贝赋值
    }
    
    std::cout << "\n--- 3. 移动语义 ---\n";
    {
        SafeTensor t1("movable", 100);
        SafeTensor t2 = std::move(t1);  // 移动构造
        SafeTensor t3("target", 50);
        t3 = std::move(t2);  // 移动赋值
    }
    
    std::cout << "\n--- 4. 禁止拷贝的类 ---\n";
    {
        NonCopyableTensor t1(100);
        // NonCopyableTensor t2 = t1;  // 编译错误！
        NonCopyableTensor t2 = std::move(t1);  // 移动OK
    }
    
    std::cout << "\n总结: 管理资源的类必须正确实现拷贝控制\n";
    return 0;
}
