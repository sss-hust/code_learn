/**
 * @file 01_rvalue_move.cpp
 * @brief 右值引用与移动语义 - 高效资源传递
 * 
 * 【推理场景】
 * - Tensor返回值优化
 * - 大数据传递避免拷贝
 * - 资源所有权转移
 * 
 * 编译: g++ -std=c++17 -o 01_rvalue_move 01_rvalue_move.cpp
 */

#include <iostream>
#include <vector>
#include <utility>

// ============================================================================
// 左值 vs 右值
// ============================================================================

void demo_lvalue_rvalue() {
    std::cout << "\n=== 左值 vs 右值 ===\n";
    
    int x = 10;        // x是左值，10是右值
    int y = x + 5;     // y是左值，x+5是右值
    
    // 左值：有名字，可以取地址
    int* px = &x;      // OK
    
    // 右值：临时值，不能取地址
    // int* p = &(x + 5);  // 错误！
    
    // 右值引用：延长临时值生命周期
    int&& rref = x + 5;  // OK
    std::cout << "右值引用: " << rref << "\n";
}

// ============================================================================
// 移动语义 Tensor
// ============================================================================

class Tensor {
private:
    float* data_;
    size_t size_;
    std::string name_;
    
public:
    // 构造
    Tensor(const std::string& name, size_t size) 
        : name_(name), size_(size) {
        data_ = new float[size_]();
        std::cout << "[构造] " << name_ << " at " << data_ << "\n";
    }
    
    // 析构
    ~Tensor() {
        std::cout << "[析构] " << name_ << " at " << data_ << "\n";
        delete[] data_;
    }
    
    // 拷贝构造（昂贵）
    Tensor(const Tensor& other) 
        : name_(other.name_ + "_copy"), size_(other.size_) {
        data_ = new float[size_];
        std::copy(other.data_, other.data_ + size_, data_);
        std::cout << "[拷贝] " << name_ << " (昂贵!)\n";
    }
    
    // 移动构造（高效）
    Tensor(Tensor&& other) noexcept 
        : data_(other.data_), size_(other.size_), 
          name_(std::move(other.name_)) {
        std::cout << "[移动] " << name_ << " (高效)\n";
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    // 移动赋值
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            name_ = std::move(other.name_);
            other.data_ = nullptr;
            other.size_ = 0;
            std::cout << "[移动赋值] " << name_ << "\n";
        }
        return *this;
    }
    
    const std::string& name() const { return name_; }
};

// 返回Tensor（触发移动或RVO）
Tensor create_tensor(const std::string& name, size_t size) {
    Tensor t(name, size);
    return t;  // 移动或编译器RVO优化
}

// 接受右值引用
void consume_tensor(Tensor&& t) {
    std::cout << "消费: " << t.name() << "\n";
    // t在函数结束后销毁
}

// ============================================================================
// std::move 的本质
// ============================================================================

void demo_std_move() {
    std::cout << "\n=== std::move ===\n";
    
    // std::move只是类型转换，将左值转为右值引用
    std::vector<float> v1(1000000, 1.0f);
    std::cout << "移动前 v1.size() = " << v1.size() << "\n";
    
    std::vector<float> v2 = std::move(v1);  // v1资源转移到v2
    std::cout << "移动后 v1.size() = " << v1.size() << "\n";
    std::cout << "v2.size() = " << v2.size() << "\n";
    
    // 注意：移动后的对象处于有效但未定义状态
    // 可以赋新值或析构，但不要使用其内容
}

// ============================================================================
// 完美转发
// ============================================================================

template<typename T>
void wrapper(T&& arg) {
    // std::forward保持参数的左/右值属性
    consume_tensor(std::forward<T>(arg));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "移动语义\n";
    std::cout << "========================================\n";
    
    demo_lvalue_rvalue();
    
    // 移动构造
    std::cout << "\n=== 移动 vs 拷贝 ===\n";
    Tensor t1("original", 1024);
    
    std::cout << "\n拷贝:\n";
    Tensor t2 = t1;  // 拷贝
    
    std::cout << "\n移动:\n";
    Tensor t3 = std::move(t1);  // 移动
    
    // 工厂函数返回
    std::cout << "\n=== 返回值优化 ===\n";
    Tensor t4 = create_tensor("created", 512);
    
    // std::move
    demo_std_move();
    
    std::cout << "\n=== 程序结束 ===\n";
    return 0;
}
