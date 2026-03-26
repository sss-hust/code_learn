/**
 * @file 02_constructors.cpp
 * @brief 构造函数与析构函数 - 资源管理的基础
 * 
 * 【推理加速场景】
 * 推理框架需要管理大量资源：
 * - GPU内存分配/释放
 * - 文件句柄（模型加载）
 * - 线程池
 * 构造/析构函数确保资源正确管理
 * 
 * 编译: g++ -std=c++17 -o 02_constructors 02_constructors.cpp
 */

#include <iostream>
#include <vector>
#include <memory>

// ============================================================================
// 第一部分：各种构造函数
// ============================================================================

class Tensor {
private:
    float* data_;
    std::vector<int> shape_;
    size_t size_;
    std::string name_;
    
public:
    // 1. 默认构造函数
    Tensor() : data_(nullptr), size_(0), name_("unnamed") {
        std::cout << "[默认构造] " << name_ << "\n";
    }
    
    // 2. 参数化构造函数
    Tensor(const std::string& name, std::vector<int> shape) 
        : name_(name), shape_(shape) {
        size_ = 1;
        for (int dim : shape_) size_ *= dim;
        data_ = new float[size_]();  // ()初始化为0
        std::cout << "[参数构造] " << name_ << ", size=" << size_ << "\n";
    }
    
    // 3. 拷贝构造函数
    Tensor(const Tensor& other) 
        : shape_(other.shape_), size_(other.size_), 
          name_(other.name_ + "_copy") {
        data_ = new float[size_];
        std::copy(other.data_, other.data_ + size_, data_);
        std::cout << "[拷贝构造] " << name_ << " from " << other.name_ << "\n";
    }
    
    // 4. 移动构造函数（C++11）
    Tensor(Tensor&& other) noexcept 
        : data_(other.data_), shape_(std::move(other.shape_)),
          size_(other.size_), name_(std::move(other.name_)) {
        other.data_ = nullptr;  // 重要：置空源对象
        other.size_ = 0;
        std::cout << "[移动构造] " << name_ << "\n";
    }
    
    // 5. 析构函数
    ~Tensor() {
        std::cout << "[析构] " << name_ << "\n";
        delete[] data_;
    }
    
    // 辅助函数
    const std::string& name() const { return name_; }
    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }
};

// ============================================================================
// 第二部分：初始化列表
// ============================================================================

class ModelConfig {
private:
    const int hidden_size_;   // const成员必须用初始化列表
    const int num_heads_;
    int& ref_value_;          // 引用成员必须用初始化列表
    
public:
    // 初始化列表：在函数体执行前初始化成员
    ModelConfig(int hidden, int heads, int& ref) 
        : hidden_size_(hidden),   // 初始化const成员
          num_heads_(heads),
          ref_value_(ref) {       // 初始化引用成员
        // 函数体中只能赋值，不能初始化
    }
    
    void print() const {
        std::cout << "hidden_size=" << hidden_size_ 
                  << ", num_heads=" << num_heads_ << "\n";
    }
};

// ============================================================================
// 第三部分：委托构造函数（C++11）
// ============================================================================

class Buffer {
private:
    float* data_;
    size_t size_;
    size_t alignment_;
    
public:
    // 主构造函数
    Buffer(size_t size, size_t alignment) 
        : size_(size), alignment_(alignment) {
        // 对齐分配
        data_ = static_cast<float*>(
            std::aligned_alloc(alignment_, size_ * sizeof(float)));
        std::cout << "[Buffer] 分配 " << size_ << " floats, 对齐=" << alignment_ << "\n";
    }
    
    // 委托构造函数：调用主构造函数
    Buffer(size_t size) : Buffer(size, 64) {  // 默认64字节对齐
        std::cout << "[Buffer] 使用默认对齐\n";
    }
    
    // 默认构造：委托
    Buffer() : Buffer(1024) {
        std::cout << "[Buffer] 使用默认大小\n";
    }
    
    ~Buffer() {
        std::free(data_);
        std::cout << "[Buffer] 释放\n";
    }
};

// ============================================================================
// 第四部分：explicit关键字
// ============================================================================

class Weight {
private:
    float value_;
    
public:
    // 不带explicit：允许隐式转换
    Weight(float v) : value_(v) {}
    
    float get() const { return value_; }
};

class SafeWeight {
private:
    float value_;
    
public:
    // explicit：禁止隐式转换
    explicit SafeWeight(float v) : value_(v) {}
    
    float get() const { return value_; }
};

void process_weight(Weight w) {
    std::cout << "处理权重: " << w.get() << "\n";
}

void process_safe_weight(SafeWeight w) {
    std::cout << "处理安全权重: " << w.get() << "\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "构造函数与析构函数\n";
    std::cout << "========================================\n";
    
    // 1. 各种构造函数
    std::cout << "\n--- 1. 各种构造函数 ---\n";
    Tensor t1;                              // 默认构造
    Tensor t2("hidden", {1, 128, 768});     // 参数构造
    Tensor t3 = t2;                         // 拷贝构造
    Tensor t4 = std::move(t3);              // 移动构造
    
    std::cout << "t3 valid: " << t3.valid() << " (被移动后为空)\n";
    std::cout << "t4 valid: " << t4.valid() << "\n";
    
    // 2. 初始化列表
    std::cout << "\n--- 2. 初始化列表 ---\n";
    int external_value = 100;
    ModelConfig config(768, 12, external_value);
    config.print();
    
    // 3. 委托构造
    std::cout << "\n--- 3. 委托构造函数 ---\n";
    Buffer buf1(2048, 32);   // 直接调用主构造
    Buffer buf2(2048);       // 委托：使用默认对齐
    Buffer buf3;             // 委托：使用默认大小和对齐
    
    // 4. explicit
    std::cout << "\n--- 4. explicit 关键字 ---\n";
    process_weight(3.14f);         // 隐式转换：float -> Weight
    // process_safe_weight(3.14f); // 错误！禁止隐式转换
    process_safe_weight(SafeWeight(3.14f));  // 必须显式构造
    
    std::cout << "\n--- 程序结束 ---\n";
    return 0;
}
