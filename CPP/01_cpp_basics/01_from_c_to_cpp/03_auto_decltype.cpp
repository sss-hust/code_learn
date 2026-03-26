/**
 * @file 03_auto_decltype.cpp
 * @brief auto 和 decltype 类型推导详解
 * 
 * 【推理加速场景】
 * 类型推导简化模板代码，与STL算法配合使用，处理复杂返回类型。
 * 
 * 编译: g++ -std=c++17 -o 03_auto_decltype 03_auto_decltype.cpp
 */

#include <iostream>
#include <vector>
#include <map>
#include <functional>
#include <cmath>

// ============================================================================
// 第一部分：auto 基础
// ============================================================================

void demo_auto_basics() {
    std::cout << "\n=== auto 基础 ===\n";
    
    // 基本类型推导
    auto i = 42;           // int
    auto d = 3.14;         // double
    auto f = 3.14f;        // float
    
    // auto与const/引用
    const int ci = 100;
    auto a1 = ci;          // int（丢失const）
    const auto a2 = ci;    // const int
    auto& a3 = ci;         // const int&（保留const）
    
    std::cout << "auto a1 = ci;       -> 丢失const\n";
    std::cout << "auto& a3 = ci;      -> 推导为const int&\n";
}

// ============================================================================
// 第二部分：auto 与容器、Lambda
// ============================================================================

void demo_auto_advanced() {
    std::cout << "\n=== auto 与容器/Lambda ===\n";
    
    // 简化容器遍历
    std::map<std::string, std::vector<float>> kv_cache;
    kv_cache["layer_0"] = {1.0f, 2.0f, 3.0f};
    
    for (const auto& [name, values] : kv_cache) {
        std::cout << name << " 有 " << values.size() << " 个元素\n";
    }
    
    // Lambda表达式（类型匿名，必须用auto）
    auto relu = [](float x) { return x > 0 ? x : 0.0f; };
    auto gelu = [](float x) { 
        return 0.5f * x * (1.0f + std::tanh(0.797885f * x));
    };
    
    std::cout << "ReLU(-1) = " << relu(-1.0f) << "\n";
    std::cout << "GELU(0.5) = " << gelu(0.5f) << "\n";
}

// ============================================================================
// 第三部分：decltype
// ============================================================================

void demo_decltype() {
    std::cout << "\n=== decltype ===\n";
    
    int x = 10;
    double y = 3.14;
    
    decltype(x) a = 20;        // int
    decltype(x + y) c = 5.5;   // double
    
    std::cout << "decltype(x+y) -> double\n";
}

// 模板中使用decltype推导返回类型
template<typename T, typename U>
auto generic_add(T a, U b) -> decltype(a + b) {
    return a + b;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "auto 和 decltype 类型推导\n";
    std::cout << "========================================\n";
    
    demo_auto_basics();
    demo_auto_advanced();
    demo_decltype();
    
    std::cout << "\n总结: auto让编译器推导类型，decltype获取精确类型\n";
    return 0;
}
