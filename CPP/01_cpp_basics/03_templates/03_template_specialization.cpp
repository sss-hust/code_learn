/**
 * @file 03_template_specialization.cpp
 * @brief 模板特化 - 针对特定类型优化
 * 
 * 【推理加速场景】
 * 为特定数据类型提供优化实现：
 * - float使用SIMD
 * - int8使用专门的量化内核
 * 
 * 编译: g++ -std=c++17 -o 03_template_specialization 03_template_specialization.cpp
 */

#include <iostream>
#include <cstdint>
#include <cstring>

// ============================================================================
// 通用模板
// ============================================================================

// 主模板：通用实现
template<typename T>
class VectorOps {
public:
    static void add(const T* a, const T* b, T* c, size_t n) {
        std::cout << "[通用实现] ";
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }
    
    static T dot(const T* a, const T* b, size_t n) {
        T sum = T(0);
        for (size_t i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
};

// ============================================================================
// 完全特化：针对float
// ============================================================================

template<>
class VectorOps<float> {
public:
    static void add(const float* a, const float* b, float* c, size_t n) {
        std::cout << "[float特化-SIMD优化] ";
        // 实际中这里会使用AVX/SSE指令
        // _mm256_add_ps 等
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }
    
    static float dot(const float* a, const float* b, size_t n) {
        // 可以使用FMA指令优化
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
};

// ============================================================================
// 完全特化：针对int8_t（量化计算）
// ============================================================================

template<>
class VectorOps<int8_t> {
public:
    static void add(const int8_t* a, const int8_t* b, int8_t* c, size_t n) {
        std::cout << "[int8特化-饱和运算] ";
        for (size_t i = 0; i < n; ++i) {
            int sum = static_cast<int>(a[i]) + static_cast<int>(b[i]);
            c[i] = static_cast<int8_t>(std::max(-128, std::min(127, sum)));
        }
    }
    
    static int32_t dot(const int8_t* a, const int8_t* b, size_t n) {
        // INT8点积通常累加到INT32
        int32_t sum = 0;
        for (size_t i = 0; i < n; ++i) {
            sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
        }
        return sum;
    }
};

// ============================================================================
// 函数模板特化
// ============================================================================

// 主模板
template<typename T>
void print_value(T val) {
    std::cout << "通用: " << val << "\n";
}

// 特化：const char*
template<>
void print_value<const char*>(const char* val) {
    std::cout << "字符串: \"" << val << "\"\n";
}

// 特化：bool
template<>
void print_value<bool>(bool val) {
    std::cout << "布尔: " << (val ? "true" : "false") << "\n";
}

// ============================================================================
// 部分特化（仅类模板支持）
// ============================================================================

// 主模板
template<typename T, int N>
class FixedArray {
    T data[N];
public:
    void info() { 
        std::cout << "FixedArray<T, " << N << ">\n"; 
    }
};

// 部分特化：指针类型
template<typename T, int N>
class FixedArray<T*, N> {
    T* data[N];
public:
    void info() { 
        std::cout << "FixedArray<T*, " << N << "> (指针特化)\n"; 
    }
};

// 部分特化：大小为1
template<typename T>
class FixedArray<T, 1> {
    T data;
public:
    void info() { 
        std::cout << "FixedArray<T, 1> (标量特化)\n"; 
    }
};

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "模板特化\n";
    std::cout << "========================================\n";
    
    // 类模板特化
    std::cout << "\n--- 类模板特化 ---\n";
    
    // 通用版本（double）
    double a_d[] = {1.0, 2.0, 3.0, 4.0};
    double b_d[] = {0.5, 0.5, 0.5, 0.5};
    double c_d[4];
    VectorOps<double>::add(a_d, b_d, c_d, 4);
    std::cout << "结果: " << c_d[0] << "\n";
    
    // float特化
    float a_f[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_f[] = {0.5f, 0.5f, 0.5f, 0.5f};
    float c_f[4];
    VectorOps<float>::add(a_f, b_f, c_f, 4);
    std::cout << "结果: " << c_f[0] << "\n";
    
    // int8特化
    int8_t a_i[] = {100, 50, -100, -50};
    int8_t b_i[] = {50, 100, -50, -100};
    int8_t c_i[4];
    VectorOps<int8_t>::add(a_i, b_i, c_i, 4);
    std::cout << "结果: " << (int)c_i[0] << " (饱和到127)\n";
    
    // 函数模板特化
    std::cout << "\n--- 函数模板特化 ---\n";
    print_value(42);
    print_value("hello");
    print_value(true);
    
    // 部分特化
    std::cout << "\n--- 部分特化 ---\n";
    FixedArray<float, 10> arr1;
    FixedArray<float*, 10> arr2;
    FixedArray<double, 1> arr3;
    
    arr1.info();
    arr2.info();
    arr3.info();
    
    return 0;
}
