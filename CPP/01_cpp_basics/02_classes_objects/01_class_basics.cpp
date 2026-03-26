/**
 * @file 01_class_basics.cpp
 * @brief 类的基础 - 封装数据和行为
 * 
 * 【推理加速场景】
 * 推理框架的核心是Tensor类，封装：
 * - 数据指针、形状、步长
 * - 设备信息（CPU/GPU）
 * - 内存管理
 * 
 * 编译: g++ -std=c++17 -o 01_class_basics 01_class_basics.cpp
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>

// ============================================================================
// 第一部分：类的基本结构
// ============================================================================

/**
 * 【对比C语言】
 * C语言使用struct + 函数指针模拟面向对象
 * C++直接支持类，更加简洁和安全
 */

// C风格（仅作对比）
struct TensorC {
    float* data;
    int* shape;
    int ndim;
};
void tensor_c_init(TensorC* t) { /* ... */ }
void tensor_c_free(TensorC* t) { /* ... */ }

// C++类
class Tensor {
// 访问控制
private:  // 私有成员，外部不可直接访问
    std::vector<float> data_;
    std::vector<int> shape_;
    std::string name_;
    
public:   // 公有成员，外部可访问
    // 构造函数
    Tensor(const std::string& name, std::vector<int> shape) 
        : name_(name), shape_(shape) {
        // 计算总元素数
        int total = 1;
        for (int dim : shape_) total *= dim;
        data_.resize(total, 0.0f);
        std::cout << "[构造] Tensor '" << name_ << "' created\n";
    }
    
    // 析构函数
    ~Tensor() {
        std::cout << "[析构] Tensor '" << name_ << "' destroyed\n";
    }
    
    // 成员函数（方法）
    size_t size() const { return data_.size(); }
    int ndim() const { return shape_.size(); }
    const std::vector<int>& shape() const { return shape_; }
    const std::string& name() const { return name_; }
    
    // 数据访问
    float& operator[](size_t i) { return data_[i]; }
    const float& operator[](size_t i) const { return data_[i]; }
    
    // 获取原始指针（与C代码交互时需要）
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    
    // 填充操作
    void fill(float value) {
        for (auto& elem : data_) elem = value;
    }
    
    // 打印信息
    void info() const {
        std::cout << "Tensor '" << name_ << "': shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "], size=" << size() << "\n";
    }
};

// ============================================================================
// 第二部分：访问控制的意义
// ============================================================================

/**
 * 【为什么需要private？】
 * 1. 封装实现细节，用户只需知道接口
 * 2. 防止外部代码破坏数据一致性
 * 3. 方便后续修改内部实现
 */

class SafeTensor {
private:
    std::vector<float> data_;
    std::vector<int> shape_;
    int total_size_;  // 缓存的总大小
    
    // 私有辅助函数
    void update_size() {
        total_size_ = 1;
        for (int dim : shape_) total_size_ *= dim;
    }
    
public:
    SafeTensor(std::vector<int> shape) : shape_(shape) {
        update_size();
        data_.resize(total_size_);
    }
    
    // 安全的reshape操作
    bool reshape(std::vector<int> new_shape) {
        int new_size = 1;
        for (int dim : new_shape) new_size *= dim;
        
        if (new_size != total_size_) {
            std::cerr << "Error: 新形状的元素数不匹配\n";
            return false;
        }
        
        shape_ = new_shape;
        return true;
    }
    
    // 如果data_和shape_是public，用户可能直接修改shape_
    // 而不更新total_size_，导致数据不一致
};

// ============================================================================
// 第三部分：静态成员
// ============================================================================

class TensorFactory {
private:
    static int tensor_count_;  // 静态成员：所有对象共享
    
public:
    static int get_count() { return tensor_count_; }
    
    static Tensor create(const std::string& name, std::vector<int> shape) {
        tensor_count_++;
        return Tensor(name, shape);
    }
};

// 静态成员需要在类外定义
int TensorFactory::tensor_count_ = 0;

// ============================================================================
// 第四部分：this指针
// ============================================================================

class Vector3D {
private:
    float x_, y_, z_;
    
public:
    Vector3D(float x, float y, float z) : x_(x), y_(y), z_(z) {}
    
    // this指针指向当前对象
    Vector3D& set_x(float x) {
        this->x_ = x;      // this->明确指定成员
        return *this;      // 返回自身引用，支持链式调用
    }
    
    Vector3D& set_y(float y) {
        this->y_ = y;
        return *this;
    }
    
    Vector3D& set_z(float z) {
        this->z_ = z;
        return *this;
    }
    
    void print() const {
        std::cout << "(" << x_ << ", " << y_ << ", " << z_ << ")\n";
    }
};

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "C++ 类的基础\n";
    std::cout << "========================================\n";
    
    // 1. 创建Tensor对象
    std::cout << "\n--- 1. 创建Tensor ---\n";
    Tensor hidden("hidden_states", {1, 128, 768});
    hidden.info();
    
    // 2. 访问成员
    std::cout << "\n--- 2. 访问成员 ---\n";
    hidden.fill(1.0f);
    std::cout << "hidden[0] = " << hidden[0] << "\n";
    
    // 3. 栈上 vs 堆上
    std::cout << "\n--- 3. 栈上 vs 堆上 ---\n";
    {
        Tensor stack_tensor("stack", {10});  // 栈上，作用域结束自动析构
    }  // 这里自动调用析构函数
    std::cout << "stack_tensor已析构\n";
    
    Tensor* heap_tensor = new Tensor("heap", {10});  // 堆上
    delete heap_tensor;  // 需要手动释放
    
    // 4. 静态成员
    std::cout << "\n--- 4. 静态成员 ---\n";
    auto t1 = TensorFactory::create("t1", {10});
    auto t2 = TensorFactory::create("t2", {20});
    std::cout << "创建的Tensor总数: " << TensorFactory::get_count() << "\n";
    
    // 5. 链式调用
    std::cout << "\n--- 5. 链式调用 ---\n";
    Vector3D vec(0, 0, 0);
    vec.set_x(1).set_y(2).set_z(3);  // 链式调用
    vec.print();
    
    std::cout << "\n--- 程序结束，自动析构 ---\n";
    return 0;
}
