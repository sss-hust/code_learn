/**
 * @file 05_inheritance.cpp
 * @brief 继承与多态 - 算子基类设计
 * 
 * 【推理加速场景】
 * 推理框架使用继承设计算子系统：
 * - Op基类定义接口
 * - 具体算子（Conv, MatMul）继承实现
 * - 多态实现统一调度
 * 
 * 编译: g++ -std=c++17 -o 05_inheritance 05_inheritance.cpp
 */

#include <iostream>
#include <vector>
#include <memory>
#include <string>

// ============================================================================
// 第一部分：算子基类
// ============================================================================

class Tensor {
public:
    std::string name;
    std::vector<float> data;
    
    Tensor(const std::string& n, size_t size) : name(n), data(size) {}
};

// 算子基类
class Operator {
protected:
    std::string name_;
    
public:
    Operator(const std::string& name) : name_(name) {}
    
    // 虚析构函数：基类必须！
    virtual ~Operator() {
        std::cout << "[~Operator] " << name_ << "\n";
    }
    
    // 纯虚函数：子类必须实现
    virtual void forward(const Tensor& input, Tensor& output) = 0;
    
    // 虚函数：可以被覆盖
    virtual std::string describe() const {
        return "Operator: " + name_;
    }
    
    const std::string& name() const { return name_; }
};

// ============================================================================
// 第二部分：具体算子
// ============================================================================

class ReLU : public Operator {
public:
    ReLU() : Operator("ReLU") {}
    
    void forward(const Tensor& input, Tensor& output) override {
        output.data.resize(input.data.size());
        for (size_t i = 0; i < input.data.size(); ++i) {
            output.data[i] = input.data[i] > 0 ? input.data[i] : 0;
        }
        std::cout << "[ReLU] forward: " << input.name << " -> " << output.name << "\n";
    }
};

class Sigmoid : public Operator {
public:
    Sigmoid() : Operator("Sigmoid") {}
    
    void forward(const Tensor& input, Tensor& output) override {
        output.data.resize(input.data.size());
        for (size_t i = 0; i < input.data.size(); ++i) {
            output.data[i] = 1.0f / (1.0f + std::exp(-input.data[i]));
        }
        std::cout << "[Sigmoid] forward\n";
    }
};

class Linear : public Operator {
private:
    int in_features_, out_features_;
    std::vector<float> weights_;
    
public:
    Linear(int in_feat, int out_feat) 
        : Operator("Linear"), in_features_(in_feat), out_features_(out_feat) {
        weights_.resize(in_feat * out_feat, 0.1f);
    }
    
    void forward(const Tensor& input, Tensor& output) override {
        // 简化的线性变换
        output.data.resize(out_features_);
        for (int o = 0; o < out_features_; ++o) {
            output.data[o] = 0;
            for (int i = 0; i < in_features_; ++i) {
                output.data[o] += input.data[i] * weights_[i * out_features_ + o];
            }
        }
        std::cout << "[Linear] " << in_features_ << " -> " << out_features_ << "\n";
    }
    
    std::string describe() const override {
        return "Linear(" + std::to_string(in_features_) + ", " + 
               std::to_string(out_features_) + ")";
    }
};

// ============================================================================
// 第三部分：多态与动态派发
// ============================================================================

class Sequential {
private:
    std::vector<std::unique_ptr<Operator>> layers_;
    
public:
    void add(std::unique_ptr<Operator> op) {
        layers_.push_back(std::move(op));
    }
    
    void forward(Tensor& x) {
        Tensor temp("temp", x.data.size());
        for (auto& layer : layers_) {
            layer->forward(x, temp);
            std::swap(x.data, temp.data);
        }
    }
    
    void describe() const {
        std::cout << "Sequential模型:\n";
        for (size_t i = 0; i < layers_.size(); ++i) {
            std::cout << "  [" << i << "] " << layers_[i]->describe() << "\n";
        }
    }
};

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "继承与多态\n";
    std::cout << "========================================\n";
    
    // 1. 多态调用
    std::cout << "\n--- 多态调用 ---\n";
    std::vector<std::unique_ptr<Operator>> ops;
    ops.push_back(std::make_unique<ReLU>());
    ops.push_back(std::make_unique<Sigmoid>());
    ops.push_back(std::make_unique<Linear>(10, 5));
    
    for (auto& op : ops) {
        std::cout << op->describe() << "\n";
    }
    
    // 2. 构建模型
    std::cout << "\n--- Sequential模型 ---\n";
    Sequential model;
    model.add(std::make_unique<Linear>(10, 20));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(20, 5));
    
    model.describe();
    
    // 3. 前向传播
    std::cout << "\n--- 前向传播 ---\n";
    Tensor input("input", 10);
    for (size_t i = 0; i < 10; ++i) input.data[i] = i * 0.1f;
    
    model.forward(input);
    
    std::cout << "\n总结: 多态是实现可扩展推理框架的关键\n";
    return 0;
}
