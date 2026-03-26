/**
 * @file 04_variadic_templates.cpp
 * @brief 可变参数模板 - 灵活的API设计
 * 
 * 【推理加速场景】
 * - 灵活的张量创建：make_tensor(1, 128, 768)
 * - 日志系统：log("batch={}, seq={}", batch, seq)
 * - 算子调用：call_kernel(kernel, arg1, arg2, ...)
 * 
 * 编译: g++ -std=c++17 -o 04_variadic_templates 04_variadic_templates.cpp
 */

#include <iostream>
#include <vector>
#include <sstream>

// ============================================================================
// 第一部分：基本可变参数模板
// ============================================================================

// 递归终止条件
void print() {
    std::cout << "\n";
}

// 递归展开
template<typename T, typename... Args>
void print(T first, Args... rest) {
    std::cout << first;
    if constexpr (sizeof...(rest) > 0) {
        std::cout << ", ";
    }
    print(rest...);  // 递归调用
}

// ============================================================================
// 第二部分：折叠表达式（C++17）
// ============================================================================

// 求和
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);  // 折叠表达式
}

// 全部满足条件
template<typename... Args>
bool all_positive(Args... args) {
    return (... && (args > 0));  // (arg1 > 0) && (arg2 > 0) && ...
}

// ============================================================================
// 第三部分：推理场景 - 创建Tensor
// ============================================================================

template<typename T>
class Tensor {
private:
    std::vector<T> data_;
    std::vector<int> shape_;
    
public:
    Tensor(std::vector<int> shape) : shape_(shape) {
        size_t size = 1;
        for (int dim : shape) size *= dim;
        data_.resize(size);
    }
    
    void info() const {
        std::cout << "Tensor: shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "], size=" << data_.size() << "\n";
    }
};

// 可变参数创建Tensor
template<typename T, typename... Dims>
Tensor<T> make_tensor(Dims... dims) {
    // 折叠表达式收集所有维度
    return Tensor<T>(std::vector<int>{static_cast<int>(dims)...});
}

// ============================================================================
// 第四部分：类型安全的printf
// ============================================================================

template<typename... Args>
std::string format(const std::string& fmt, Args... args) {
    std::ostringstream oss;
    
    // 将参数存入数组
    std::vector<std::string> arg_strs;
    (arg_strs.push_back([](auto arg) {
        std::ostringstream s;
        s << arg;
        return s.str();
    }(args)), ...);
    
    size_t arg_idx = 0;
    for (size_t i = 0; i < fmt.size(); ++i) {
        if (fmt[i] == '{' && i + 1 < fmt.size() && fmt[i + 1] == '}') {
            if (arg_idx < arg_strs.size()) {
                oss << arg_strs[arg_idx++];
            }
            ++i;
        } else {
            oss << fmt[i];
        }
    }
    return oss.str();
}

// ============================================================================
// 第五部分：参数包大小和转发
// ============================================================================

template<typename... Args>
void show_pack_info(Args... args) {
    std::cout << "参数数量: " << sizeof...(Args) << "\n";
    std::cout << "参数值: ";
    print(args...);
}

// 完美转发
template<typename F, typename... Args>
auto invoke(F&& f, Args&&... args) {
    return std::forward<F>(f)(std::forward<Args>(args)...);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "可变参数模板\n";
    std::cout << "========================================\n";
    
    // 基本打印
    std::cout << "\n--- 可变参数打印 ---\n";
    print(1, 2.5, "hello", 'x');
    
    // 折叠表达式
    std::cout << "\n--- 折叠表达式 ---\n";
    std::cout << "sum(1,2,3,4,5) = " << sum(1, 2, 3, 4, 5) << "\n";
    std::cout << "all_positive(1,2,3) = " << all_positive(1, 2, 3) << "\n";
    std::cout << "all_positive(1,-2,3) = " << all_positive(1, -2, 3) << "\n";
    
    // 创建Tensor
    std::cout << "\n--- 创建Tensor ---\n";
    auto t1 = make_tensor<float>(1, 128, 768);  // batch=1, seq=128, hidden=768
    auto t2 = make_tensor<float>(32, 12, 128, 64);  // batch, heads, seq, head_dim
    t1.info();
    t2.info();
    
    // 格式化字符串
    std::cout << "\n--- 格式化 ---\n";
    std::string msg = format("batch={}, seq_len={}, hidden={}", 32, 128, 768);
    std::cout << msg << "\n";
    
    // 参数包信息
    std::cout << "\n--- 参数包信息 ---\n";
    show_pack_info(1, 2.0f, "三");
    
    // 调用函数
    std::cout << "\n--- 完美转发 ---\n";
    auto add = [](int a, int b) { return a + b; };
    std::cout << "invoke(add, 3, 4) = " << invoke(add, 3, 4) << "\n";
    
    return 0;
}
