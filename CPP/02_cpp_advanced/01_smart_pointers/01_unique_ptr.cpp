/**
 * @file 01_unique_ptr.cpp
 * @brief unique_ptr - 独占所有权智能指针
 * 
 * 【推理场景】
 * - 独占资源管理（GPU buffer）
 * - 工厂模式创建对象
 * - 多态容器
 * 
 * 编译: g++ -std=c++17 -o 01_unique_ptr 01_unique_ptr.cpp
 */

#include <iostream>
#include <memory>
#include <vector>

// 模拟Tensor类
class Tensor {
private:
    std::vector<float> data_;
    std::string name_;
    
public:
    Tensor(const std::string& name, size_t size) 
        : name_(name), data_(size) {
        std::cout << "[创建] " << name_ << "\n";
    }
    
    ~Tensor() {
        std::cout << "[销毁] " << name_ << "\n";
    }
    
    const std::string& name() const { return name_; }
    size_t size() const { return data_.size(); }
};

// 算子基类
class Operator {
public:
    virtual ~Operator() = default;
    virtual void forward() = 0;
};

class ReLU : public Operator {
public:
    void forward() override { std::cout << "ReLU forward\n"; }
};

class Sigmoid : public Operator {
public:
    void forward() override { std::cout << "Sigmoid forward\n"; }
};

int main() {
    std::cout << "========================================\n";
    std::cout << "unique_ptr 智能指针\n";
    std::cout << "========================================\n";
    
    // 1. 基本用法
    std::cout << "\n--- 基本用法 ---\n";
    {
        std::unique_ptr<Tensor> t1 = std::make_unique<Tensor>("hidden", 1024);
        std::cout << "使用: " << t1->name() << ", size=" << t1->size() << "\n";
    }  // 自动释放，无需delete
    
    // 2. 所有权转移
    std::cout << "\n--- 所有权转移 ---\n";
    auto t2 = std::make_unique<Tensor>("weights", 768);
    std::cout << "t2有效: " << (t2 != nullptr) << "\n";
    
    auto t3 = std::move(t2);  // 转移所有权
    std::cout << "转移后t2有效: " << (t2 != nullptr) << "\n";
    std::cout << "t3有效: " << (t3 != nullptr) << "\n";
    
    // 3. 多态容器
    std::cout << "\n--- 多态容器 ---\n";
    std::vector<std::unique_ptr<Operator>> layers;
    layers.push_back(std::make_unique<ReLU>());
    layers.push_back(std::make_unique<Sigmoid>());
    layers.push_back(std::make_unique<ReLU>());
    
    for (auto& layer : layers) {
        layer->forward();
    }
    
    // 4. 工厂函数
    std::cout << "\n--- 工厂函数 ---\n";
    auto create_op = [](const std::string& type) -> std::unique_ptr<Operator> {
        if (type == "relu") return std::make_unique<ReLU>();
        if (type == "sigmoid") return std::make_unique<Sigmoid>();
        return nullptr;
    };
    
    auto op = create_op("relu");
    if (op) op->forward();
    
    // 5. 释放原始指针（与C API交互）
    std::cout << "\n--- 释放原始指针 ---\n";
    auto t4 = std::make_unique<Tensor>("temp", 100);
    Tensor* raw = t4.release();  // unique_ptr放弃所有权
    std::cout << "手动管理: " << raw->name() << "\n";
    delete raw;  // 必须手动删除
    
    std::cout << "\n--- 程序结束 ---\n";
    return 0;
}
