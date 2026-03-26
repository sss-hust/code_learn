/**
 * @file 02_class_templates.cpp
 * @brief 类模板 - 多精度Tensor实现
 * 
 * 【推理加速场景】
 * 支持多种数据类型的Tensor：
 * - Tensor<float> FP32推理
 * - Tensor<half> FP16推理
 * - Tensor<int8_t> INT8量化推理
 * 
 * 编译: g++ -std=c++17 -o 02_class_templates 02_class_templates.cpp
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <type_traits>

// ============================================================================
// 多精度Tensor类模板
// ============================================================================

template<typename T>
class Tensor {
private:
    std::vector<T> data_;
    std::vector<int> shape_;
    
public:
    // 类型别名
    using value_type = T;
    using size_type = size_t;
    
    // 构造函数
    Tensor(std::vector<int> shape, T init_val = T(0)) : shape_(shape) {
        size_t total = 1;
        for (int dim : shape) total *= dim;
        data_.resize(total, init_val);
    }
    
    // 访问器
    size_t size() const { return data_.size(); }
    const std::vector<int>& shape() const { return shape_; }
    
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    
    // 填充
    void fill(T value) {
        for (auto& elem : data_) elem = value;
    }
    
    // 获取数据类型大小
    static constexpr size_t dtype_size() { return sizeof(T); }
    
    // 计算内存占用
    size_t memory_bytes() const { return size() * sizeof(T); }
    
    // 打印信息
    void info() const {
        std::cout << "Tensor<" << sizeof(T) << " bytes>: shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "], memory=" << memory_bytes() / 1024.0 << " KB\n";
    }
};

// ============================================================================
// 类型转换
// ============================================================================

// 量化：float -> int8
template<typename T>
Tensor<int8_t> quantize(const Tensor<T>& input, T scale) {
    Tensor<int8_t> output(input.shape());
    for (size_t i = 0; i < input.size(); ++i) {
        int val = static_cast<int>(input[i] / scale);
        val = std::max(-128, std::min(127, val));
        output[i] = static_cast<int8_t>(val);
    }
    return output;
}

// 反量化：int8 -> float
template<typename T = float>
Tensor<T> dequantize(const Tensor<int8_t>& input, T scale) {
    Tensor<T> output(input.shape());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<T>(input[i]) * scale;
    }
    return output;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "类模板 - 多精度Tensor\n";
    std::cout << "========================================\n";
    
    // 不同精度的Tensor
    std::cout << "\n--- 不同精度 ---\n";
    Tensor<float> fp32_tensor({1, 128, 768});
    Tensor<double> fp64_tensor({1, 128, 768});
    Tensor<int8_t> int8_tensor({1, 128, 768});
    
    fp32_tensor.info();
    fp64_tensor.info();
    int8_tensor.info();
    
    // 量化示例
    std::cout << "\n--- 量化 ---\n";
    Tensor<float> weights({4}, 0.0f);
    weights[0] = 0.5f;
    weights[1] = -0.3f;
    weights[2] = 1.2f;
    weights[3] = -0.8f;
    
    float scale = 0.01f;
    auto quantized = quantize(weights, scale);
    
    std::cout << "原始: [" << weights[0] << ", " << weights[1] << ", ...]\n";
    std::cout << "量化: [" << (int)quantized[0] << ", " << (int)quantized[1] << ", ...]\n";
    
    auto recovered = dequantize(quantized, scale);
    std::cout << "恢复: [" << recovered[0] << ", " << recovered[1] << ", ...]\n";
    
    // 内存节省
    std::cout << "\n--- 内存节省 ---\n";
    std::cout << "FP32: " << fp32_tensor.memory_bytes() / 1024.0 / 1024.0 << " MB\n";
    std::cout << "INT8: " << int8_tensor.memory_bytes() / 1024.0 / 1024.0 << " MB\n";
    std::cout << "节省: " << (1.0 - float(int8_tensor.memory_bytes()) / fp32_tensor.memory_bytes()) * 100 << "%\n";
    
    return 0;
}
