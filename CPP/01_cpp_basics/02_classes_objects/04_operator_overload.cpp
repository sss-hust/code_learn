/**
 * @file 04_operator_overload.cpp
 * @brief 运算符重载 - 让自定义类型像内置类型一样使用
 * 
 * 【推理加速场景】
 * Tensor运算符重载使代码更直观：
 * - C = A + B 代替 add(A, B, C)
 * - tensor[i] 代替 tensor.at(i)
 * 
 * 编译: g++ -std=c++17 -o 04_operator_overload 04_operator_overload.cpp
 */

#include <iostream>
#include <vector>
#include <cassert>

// ============================================================================
// Tensor类运算符重载示例
// ============================================================================

class Tensor {
private:
    std::vector<float> data_;
    std::vector<int> shape_;
    
public:
    Tensor(std::vector<int> shape, float init_val = 0.0f) 
        : shape_(shape) {
        size_t size = 1;
        for (int dim : shape) size *= dim;
        data_.resize(size, init_val);
    }
    
    size_t size() const { return data_.size(); }
    const std::vector<int>& shape() const { return shape_; }
    
    // ========== 下标运算符 ==========
    float& operator[](size_t i) { return data_[i]; }
    const float& operator[](size_t i) const { return data_[i]; }
    
    // ========== 算术运算符 ==========
    
    // 加法：Tensor + Tensor
    Tensor operator+(const Tensor& other) const {
        assert(size() == other.size());
        Tensor result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data_[i] + other.data_[i];
        }
        return result;
    }
    
    // 减法
    Tensor operator-(const Tensor& other) const {
        assert(size() == other.size());
        Tensor result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data_[i] - other.data_[i];
        }
        return result;
    }
    
    // 标量乘法：Tensor * scalar
    Tensor operator*(float scalar) const {
        Tensor result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data_[i] * scalar;
        }
        return result;
    }
    
    // ========== 复合赋值运算符 ==========
    
    Tensor& operator+=(const Tensor& other) {
        assert(size() == other.size());
        for (size_t i = 0; i < size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }
    
    Tensor& operator*=(float scalar) {
        for (size_t i = 0; i < size(); ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }
    
    // ========== 比较运算符 ==========
    
    bool operator==(const Tensor& other) const {
        if (shape_ != other.shape_) return false;
        for (size_t i = 0; i < size(); ++i) {
            if (data_[i] != other.data_[i]) return false;
        }
        return true;
    }
    
    bool operator!=(const Tensor& other) const {
        return !(*this == other);
    }
    
    // ========== 流输出运算符（友元函数）==========
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << "Tensor(shape=[";
        for (size_t i = 0; i < t.shape_.size(); ++i) {
            os << t.shape_[i];
            if (i < t.shape_.size() - 1) os << ", ";
        }
        os << "], data=[";
        size_t max_print = std::min(t.size(), size_t(5));
        for (size_t i = 0; i < max_print; ++i) {
            os << t.data_[i];
            if (i < max_print - 1) os << ", ";
        }
        if (t.size() > 5) os << ", ...";
        os << "])";
        return os;
    }
};

// 标量 * Tensor（非成员函数）
Tensor operator*(float scalar, const Tensor& t) {
    return t * scalar;  // 复用成员函数
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "运算符重载\n";
    std::cout << "========================================\n";
    
    // 创建Tensor
    Tensor a({4}, 1.0f);
    Tensor b({4}, 2.0f);
    
    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";
    
    // 算术运算
    std::cout << "\n--- 算术运算 ---\n";
    Tensor c = a + b;
    std::cout << "a + b = " << c << "\n";
    
    Tensor d = a * 3.0f;
    std::cout << "a * 3 = " << d << "\n";
    
    Tensor e = 2.0f * b;
    std::cout << "2 * b = " << e << "\n";
    
    // 复合赋值
    std::cout << "\n--- 复合赋值 ---\n";
    a += b;
    std::cout << "a += b: " << a << "\n";
    
    // 下标访问
    std::cout << "\n--- 下标访问 ---\n";
    a[0] = 100.0f;
    std::cout << "a[0] = " << a[0] << "\n";
    
    // 比较
    std::cout << "\n--- 比较运算 ---\n";
    Tensor f({4}, 3.0f);
    Tensor g({4}, 3.0f);
    std::cout << "f == g: " << (f == g) << "\n";
    
    return 0;
}
