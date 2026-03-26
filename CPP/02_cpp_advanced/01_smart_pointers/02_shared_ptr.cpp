/**
 * @file 02_shared_ptr.cpp
 * @brief shared_ptr - 共享所有权智能指针
 * 
 * 【推理场景】
 * - 模型权重共享（多个层共享embedding）
 * - 缓存数据共享
 * - 异步任务间共享资源
 * 
 * 编译: g++ -std=c++17 -o 02_shared_ptr 02_shared_ptr.cpp
 */

#include <iostream>
#include <memory>
#include <vector>

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
};

int main() {
    std::cout << "========================================\n";
    std::cout << "shared_ptr 智能指针\n";
    std::cout << "========================================\n";
    
    // 1. 基本用法
    std::cout << "\n--- 基本用法 ---\n";
    auto t1 = std::make_shared<Tensor>("embedding", 50257 * 768);
    std::cout << "引用计数: " << t1.use_count() << "\n";
    
    // 2. 共享所有权
    std::cout << "\n--- 共享所有权 ---\n";
    {
        auto t2 = t1;  // 拷贝，共享所有权
        std::cout << "t2 = t1后，引用计数: " << t1.use_count() << "\n";
        
        auto t3 = t1;
        std::cout << "t3 = t1后，引用计数: " << t1.use_count() << "\n";
    }  // t2, t3离开作用域
    std::cout << "t2,t3销毁后，引用计数: " << t1.use_count() << "\n";
    
    // 3. 权重共享场景
    std::cout << "\n--- 权重共享 ---\n";
    struct TransformerBlock {
        std::shared_ptr<Tensor> embedding;
        
        TransformerBlock(std::shared_ptr<Tensor> emb) 
            : embedding(emb) {}
    };
    
    auto shared_embedding = std::make_shared<Tensor>("shared_emb", 1000);
    std::cout << "共享前引用计数: " << shared_embedding.use_count() << "\n";
    
    std::vector<TransformerBlock> layers;
    for (int i = 0; i < 12; ++i) {
        layers.emplace_back(shared_embedding);
    }
    std::cout << "12层共享后引用计数: " << shared_embedding.use_count() << "\n";
    
    // 4. weak_ptr 避免循环引用
    std::cout << "\n--- weak_ptr ---\n";
    std::weak_ptr<Tensor> weak = t1;
    std::cout << "weak_ptr不增加引用计数: " << t1.use_count() << "\n";
    
    if (auto locked = weak.lock()) {  // 尝试获取shared_ptr
        std::cout << "资源仍有效: " << locked->name() << "\n";
    }
    
    t1.reset();  // 释放
    std::cout << "t1释放后，weak.expired(): " << weak.expired() << "\n";
    
    std::cout << "\n--- 程序结束 ---\n";
    return 0;
}
