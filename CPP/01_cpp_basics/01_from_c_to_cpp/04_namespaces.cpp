/**
 * @file 04_namespaces.cpp
 * @brief 命名空间详解 - 模块化代码组织
 * 
 * 【推理加速场景】
 * 大型推理框架使用命名空间组织代码：
 * - torch::Tensor, at::native 等
 * - 避免符号冲突，区分不同模块
 * 
 * 编译: g++ -std=c++17 -o 04_namespaces 04_namespaces.cpp
 */

#include <iostream>
#include <vector>

// ============================================================================
// 第一部分：基本命名空间
// ============================================================================

namespace inference {
    const int VERSION = 1;
    
    void initialize() {
        std::cout << "初始化推理引擎 v" << VERSION << "\n";
    }
    
    // 嵌套命名空间
    namespace cpu {
        void run() { std::cout << "CPU推理\n"; }
    }
    
    namespace gpu {
        void run() { std::cout << "GPU推理\n"; }
    }
}

// C++17 嵌套命名空间简化语法
namespace model::attention {
    void forward() { std::cout << "Attention前向\n"; }
}

// ============================================================================
// 第二部分：避免命名冲突
// ============================================================================

namespace framework_a {
    class Tensor {
    public:
        void info() { std::cout << "Framework A的Tensor\n"; }
    };
}

namespace framework_b {
    class Tensor {
    public:
        void info() { std::cout << "Framework B的Tensor\n"; }
    };
}

// ============================================================================
// 第三部分：using声明
// ============================================================================

void demo_using() {
    std::cout << "\n=== using声明 ===\n";
    
    // using声明：引入特定名称
    using inference::initialize;
    initialize();  // 不用写inference::
    
    // using指令：引入整个命名空间（谨慎使用）
    using namespace inference::cpu;
    run();  // 直接调用
    
    // 类型别名
    using FloatVec = std::vector<float>;
    FloatVec data = {1.0f, 2.0f, 3.0f};
}

int main() {
    std::cout << "========================================\n";
    std::cout << "命名空间详解\n";
    std::cout << "========================================\n";
    
    // 完全限定名
    inference::initialize();
    inference::cpu::run();
    inference::gpu::run();
    
    // 嵌套命名空间
    model::attention::forward();
    
    // 避免命名冲突
    std::cout << "\n=== 避免命名冲突 ===\n";
    framework_a::Tensor ta;
    framework_b::Tensor tb;
    ta.info();
    tb.info();
    
    demo_using();
    
    std::cout << "\n总结: 命名空间是组织大型项目的关键\n";
    return 0;
}
