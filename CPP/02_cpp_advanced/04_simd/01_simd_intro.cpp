/**
 * @file 01_simd_intro.cpp
 * @brief SIMD向量化 - 单指令多数据
 * 
 * 【推理场景】
 * - 矩阵运算加速
 * - 批量激活函数
 * - 向量点积
 * 
 * 编译: g++ -std=c++17 -O2 -mavx2 -o 01_simd 01_simd_intro.cpp
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// 如果有AVX支持，取消注释
// #include <immintrin.h>

// ============================================================================
// 标量版本（朴素实现）
// ============================================================================

void vector_add_scalar(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void relu_scalar(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}

float dot_product_scalar(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ============================================================================
// 手动展开优化（帮助编译器向量化）
// ============================================================================

void vector_add_unrolled(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    // 每次处理8个元素
    for (; i + 7 < n; i += 8) {
        c[i]   = a[i]   + b[i];
        c[i+1] = a[i+1] + b[i+1];
        c[i+2] = a[i+2] + b[i+2];
        c[i+3] = a[i+3] + b[i+3];
        c[i+4] = a[i+4] + b[i+4];
        c[i+5] = a[i+5] + b[i+5];
        c[i+6] = a[i+6] + b[i+6];
        c[i+7] = a[i+7] + b[i+7];
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================================
// AVX Intrinsics版本（需要AVX支持）
// ============================================================================

#ifdef __AVX__
#include <immintrin.h>

void vector_add_avx(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    // AVX一次处理8个float
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&c[i], vc);
    }
    // 处理剩余
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void relu_avx(float* data, size_t n) {
    __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        v = _mm256_max_ps(v, zero);  // max(v, 0)
        _mm256_storeu_ps(&data[i], v);
    }
    for (; i < n; ++i) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}

float dot_product_avx(const float* a, const float* b, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);  // sum += a * b
    }
    
    // 水平求和
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);
    
    // 处理剩余
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif

// ============================================================================
// 性能测试
// ============================================================================

template<typename Func>
double benchmark(Func f, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        f();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "========================================\n";
    std::cout << "SIMD向量化基础\n";
    std::cout << "========================================\n";
    
    const size_t N = 1000000;
    const int ITER = 100;
    
    // 准备数据
    std::vector<float> a(N), b(N), c(N);
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i) * 0.001f;
        b[i] = static_cast<float>(N - i) * 0.001f;
    }
    
    // 标量版本
    std::cout << "\n--- 性能测试 ---\n";
    
    double scalar_time = benchmark([&]() {
        vector_add_scalar(a.data(), b.data(), c.data(), N);
    }, ITER);
    std::cout << "标量版本: " << scalar_time << " ms\n";
    
    double unrolled_time = benchmark([&]() {
        vector_add_unrolled(a.data(), b.data(), c.data(), N);
    }, ITER);
    std::cout << "展开版本: " << unrolled_time << " ms\n";
    
#ifdef __AVX__
    double avx_time = benchmark([&]() {
        vector_add_avx(a.data(), b.data(), c.data(), N);
    }, ITER);
    std::cout << "AVX版本: " << avx_time << " ms\n";
    std::cout << "加速比: " << scalar_time / avx_time << "x\n";
#else
    std::cout << "(AVX未启用，使用-mavx2编译以启用)\n";
#endif
    
    // 点积测试
    std::cout << "\n--- 点积 ---\n";
    float result = dot_product_scalar(a.data(), b.data(), N);
    std::cout << "点积结果: " << result << "\n";
    
    // SIMD概念说明
    std::cout << "\n=== SIMD原理 ===\n";
    std::cout << "标量: 一次处理1个元素\n";
    std::cout << "SSE:  一次处理4个float (128位)\n";
    std::cout << "AVX:  一次处理8个float (256位)\n";
    std::cout << "AVX-512: 一次处理16个float (512位)\n";
    
    return 0;
}
