/**
 * @file 05_overloading.cpp
 * @brief 函数重载详解 - 统一接口，多种实现
 * 
 * 【推理加速场景】
 * 重载用于处理不同数据类型和形状：
 * - matmul(float*, float*) vs matmul(int8_t*, int8_t*)
 * - forward(Tensor) vs forward(Tensor, Tensor)
 * 
 * 编译: g++ -std=c++17 -o 05_overloading 05_overloading.cpp
 */

#include <iostream>
#include <vector>
#include <cstdint>

// ============================================================================
// 第一部分：基本函数重载
// ============================================================================

// 同名函数，不同参数类型
void process(int x) {
    std::cout << "处理int: " << x << "\n";
}

void process(float x) {
    std::cout << "处理float: " << x << "\n";
}

void process(const std::vector<float>& v) {
    std::cout << "处理vector，大小: " << v.size() << "\n";
}

// ============================================================================
// 第二部分：推理场景 - 多精度计算
// ============================================================================

namespace compute {
    
// FP32矩阵乘法
void matmul(const float* A, const float* B, float* C, 
            int M, int N, int K) {
    std::cout << "FP32 MatMul: " << M << "x" << K << " * " << K << "x" << N << "\n";
    // 实际计算省略
}

// INT8量化矩阵乘法
void matmul(const int8_t* A, const int8_t* B, int32_t* C,
            int M, int N, int K) {
    std::cout << "INT8 MatMul: " << M << "x" << K << " * " << K << "x" << N << "\n";
    // INT8计算通常更快
}

// 带bias的矩阵乘法
void matmul(const float* A, const float* B, const float* bias, float* C,
            int M, int N, int K) {
    std::cout << "FP32 MatMul + Bias\n";
}

} // namespace compute

// ============================================================================
// 第三部分：运算符重载预告
// ============================================================================

class SimpleVec {
public:
    float x, y, z;
    
    SimpleVec(float x, float y, float z) : x(x), y(y), z(z) {}
    
    // 运算符重载：向量加法
    SimpleVec operator+(const SimpleVec& other) const {
        return SimpleVec(x + other.x, y + other.y, z + other.z);
    }
    
    void print() const {
        std::cout << "(" << x << ", " << y << ", " << z << ")\n";
    }
};

int main() {
    std::cout << "========================================\n";
    std::cout << "函数重载详解\n";
    std::cout << "========================================\n";
    
    // 基本重载
    std::cout << "\n--- 基本重载 ---\n";
    process(42);
    process(3.14f);
    process(std::vector<float>{1.0f, 2.0f});
    
    // 多精度计算
    std::cout << "\n--- 多精度计算 ---\n";
    float A[4], B[4], C[4], bias[2];
    int8_t Aq[4], Bq[4];
    int32_t Cq[4];
    
    compute::matmul(A, B, C, 2, 2, 2);        // FP32
    compute::matmul(Aq, Bq, Cq, 2, 2, 2);     // INT8
    compute::matmul(A, B, bias, C, 2, 2, 2);  // FP32 + bias
    
    // 运算符重载
    std::cout << "\n--- 运算符重载 ---\n";
    SimpleVec v1(1, 2, 3);
    SimpleVec v2(4, 5, 6);
    SimpleVec v3 = v1 + v2;
    std::cout << "v1 + v2 = ";
    v3.print();
    
    return 0;
}
