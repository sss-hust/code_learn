/**
 * @file 01_function_templates.cpp
 * @brief 函数模板 - 泛型编程基础
 * 
 * 【推理加速场景】
 * 同一算法支持多种数据类型：
 * - FP32/FP16/INT8 计算
 * - 避免代码重复
 * 
 * 编译: g++ -std=c++17 -o 01_function_templates 01_function_templates.cpp
 */

#include <iostream>
#include <vector>
#include <cmath>

// ============================================================================
// 第一部分：基本函数模板
// ============================================================================

// 模板函数：适用于任何类型
template<typename T>
T add(T a, T b) {
    return a + b;
}

// 多类型参数
template<typename T, typename U>
auto multiply(T a, U b) -> decltype(a * b) {
    return a * b;
}

// ============================================================================
// 第二部分：推理场景 - 通用向量运算
// ============================================================================

// 向量加法
template<typename T>
void vector_add(const T* a, const T* b, T* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// 向量缩放
template<typename T>
void vector_scale(T* data, T scale, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] *= scale;
    }
}

// ReLU激活函数
template<typename T>
void relu(T* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (data[i] < T(0)) data[i] = T(0);
    }
}

// Softmax
template<typename T>
void softmax(T* data, size_t n) {
    T max_val = data[0];
    for (size_t i = 1; i < n; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    T sum = T(0);
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    
    for (size_t i = 0; i < n; ++i) {
        data[i] /= sum;
    }
}

// ============================================================================
// 第三部分：模板参数推导
// ============================================================================

// 编译器自动推导T
template<typename T>
void print_type(T value) {
    std::cout << "值: " << value << ", 类型大小: " << sizeof(T) << " bytes\n";
}

// 容器元素类型推导
template<typename Container>
auto compute_sum(const Container& c) {
    using T = typename Container::value_type;
    T sum = T(0);
    for (const auto& val : c) sum += val;
    return sum;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "函数模板\n";
    std::cout << "========================================\n";
    
    // 基本模板
    std::cout << "\n--- 基本模板 ---\n";
    std::cout << "add(1, 2) = " << add(1, 2) << "\n";
    std::cout << "add(1.5, 2.5) = " << add(1.5, 2.5) << "\n";
    std::cout << "multiply(3, 1.5) = " << multiply(3, 1.5) << "\n";
    
    // 向量运算
    std::cout << "\n--- 向量运算 ---\n";
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {0.5f, 0.5f, 0.5f, 0.5f};
    float c[4];
    
    vector_add(a, b, c, 4);
    std::cout << "vector_add: [" << c[0] << ", " << c[1] << ", ...]\n";
    
    // 多精度支持
    std::cout << "\n--- 多精度支持 ---\n";
    double d_arr[] = {-1.0, 2.0, -3.0, 4.0};
    relu(d_arr, 4);
    std::cout << "ReLU(double): [" << d_arr[0] << ", " << d_arr[1] << ", ...]\n";
    
    // Softmax
    float logits[] = {1.0f, 2.0f, 3.0f};
    softmax(logits, 3);
    std::cout << "Softmax: [" << logits[0] << ", " << logits[1] << ", " << logits[2] << "]\n";
    
    // 类型推导
    std::cout << "\n--- 类型推导 ---\n";
    print_type(42);
    print_type(3.14f);
    print_type(3.14);
    
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    std::cout << "sum = " << compute_sum(vec) << "\n";
    
    return 0;
}
