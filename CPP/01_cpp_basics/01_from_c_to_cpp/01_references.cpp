/**
 * @file 01_references.cpp
 * @brief C++ 引用详解 - 从C指针到C++引用的过渡
 * 
 * 【推理加速场景】
 * 在大模型推理中，Tensor数据通常很大（GB级别），传递时使用引用可以：
 * 1. 避免不必要的数据拷贝，节省内存和时间
 * 2. 使代码更清晰，避免指针的复杂性
 * 3. 保证引用始终有效（不能为null）
 * 
 * 编译: g++ -std=c++17 -o 01_references 01_references.cpp
 * 运行: ./01_references
 */

#include <iostream>
#include <vector>

// ============================================================================
// 第一部分：引用基础
// ============================================================================

/**
 * 【C语言回顾】使用指针交换两个值
 * 问题：需要解引用，容易出错，可能传入空指针
 */
void swap_c_style(int* a, int* b) {
    if (a == nullptr || b == nullptr) return;  // 必须检查空指针
    int temp = *a;  // 需要解引用
    *a = *b;
    *b = temp;
}

/**
 * 【C++改进】使用引用交换两个值
 * 优点：
 * 1. 不需要解引用，代码更清晰
 * 2. 引用不能为null，无需检查
 * 3. 调用时不需要取地址符&
 */
// 相当于是封装好的语法糖，引用是一个别名
void swap_cpp_style(int& a, int& b) {
    int temp = a;  // 直接使用，无需解引用
    a = b;
    b = temp;
}

// ============================================================================
// 第二部分：const 引用 - 推理加速的关键
// ============================================================================

/**
 * 【场景】模拟一个简单的Tensor结构
 * 在实际推理框架中，Tensor可能包含数百MB甚至GB的数据
 */
struct Tensor {
    std::vector<float> data;      // 数据存储
    std::vector<int> shape;       // 形状 [batch, seq_len, hidden_dim]
    std::string name;             // 张量名称
    
    // 构造函数
    Tensor(const std::string& n, std::vector<int> s) 
        : name(n), shape(s) {
        // 计算总元素数并分配内存
        int total = 1;
        for (int dim : shape) total *= dim;
        data.resize(total, 0.0f);
        std::cout << "  [构造] Tensor '" << name << "' 创建，大小: " 
                  << total * sizeof(float) / 1024.0 / 1024.0 << " MB\n";
    }
    
    // 拷贝构造函数 - 用于演示拷贝开销
    Tensor(const Tensor& other) 
        : data(other.data), shape(other.shape), name(other.name + "_copy") {
        std::cout << "  [拷贝] Tensor 被拷贝! 这很慢!\n";
    }
};

/**
 * 【错误示范】按值传递 - 会触发拷贝
 * 问题：每次调用都会拷贝整个Tensor，对于大Tensor是灾难性的
 */
void process_tensor_by_value(Tensor t) {
    std::cout << "  处理Tensor (按值): " << t.name << "\n";
    // 这里会自动拷贝整个Tensor！
}

/**
 * 【正确做法】按const引用传递 - 无拷贝
 * 优点：
 * 1. 零拷贝：直接访问原始数据
 * 2. 安全：const保证不会修改原数据
 * 3. 清晰：比指针更直观
 */
void process_tensor_by_ref(const Tensor& t) {
    std::cout << "  处理Tensor (const引用): " << t.name << "\n";
    // 直接访问原始数据，无拷贝
}

/**
 * 【需要修改时】使用非const引用
 * 场景：原地操作（in-place operation），节省内存
 */
void relu_inplace(Tensor& t) {
    std::cout << "  原地ReLU操作: " << t.name << "\n";
    for (float& val : t.data) {
        if (val < 0) val = 0;
    }
}

// ============================================================================
// 第三部分：引用作为返回值
// ============================================================================

/**
 * 【场景】模拟KV Cache中的缓存查找
 * 返回引用避免拷贝，同时允许直接修改缓存
 */
class KVCache {
private:
    std::vector<Tensor> cache;
    
public:
    // 添加新的缓存项
    void add(const std::string& name, std::vector<int> shape) {
        cache.emplace_back(name, shape);
    }
    
    /**
     * 返回引用：允许直接访问和修改缓存中的Tensor
     * 注意：返回引用时要确保对象的生命周期足够长
     */
    Tensor& get(size_t index) {
        if (index >= cache.size()) {
            throw std::out_of_range("缓存索引越界");
        }
        return cache[index];  // 返回引用，不拷贝
    }
    
    // 返回const引用：只读访问
    const Tensor& get_readonly(size_t index) const {
        return cache[index];
    }
    
    size_t size() const { return cache.size(); }
};

// ============================================================================
// 第四部分：引用 vs 指针 对比
// ============================================================================

void demo_reference_vs_pointer() {
    std::cout << "\n=== 引用 vs 指针 对比 ===\n";
    
    int value = 42;
    
    // 指针方式
    int* ptr = &value;      // 需要取地址
    *ptr = 100;             // 需要解引用
    ptr = nullptr;          // 可以重新指向null
    
    // 引用方式
    int& ref = value;       // 直接初始化
    ref = 200;              // 直接使用
    // ref = nullptr;       // 错误！引用不能为null
    // int& ref2;           // 错误！引用必须初始化
    
    std::cout << "引用的优势:\n";
    std::cout << "  1. 语法更简洁（无需*和&操作符）\n";
    std::cout << "  2. 更安全（不能为null，必须初始化）\n";
    std::cout << "  3. 不能重新绑定（一旦绑定，始终指向同一对象）\n";
    
    std::cout << "指针的必要场景:\n";
    std::cout << "  1. 需要重新指向不同对象\n";
    std::cout << "  2. 需要表示"空"状态\n";
    std::cout << "  3. 与C代码交互\n";
}

// ============================================================================
// 第五部分：实际推理场景示例
// ============================================================================

/**
 * 【实际场景】模拟Transformer层的前向传播
 * 使用const引用传入，使用非const引用输出
 */
void transformer_layer_forward(
    const Tensor& input,      // 输入：只读
    const Tensor& weights,    // 权重：只读
    Tensor& output            // 输出：可写
) {
    std::cout << "  Transformer层前向传播:\n";
    std::cout << "    输入: " << input.name << "\n";
    std::cout << "    权重: " << weights.name << "\n";
    std::cout << "    输出: " << output.name << "\n";
    
    // 实际计算会在这里进行
    // 使用引用避免了任何不必要的拷贝
}

// ============================================================================
// 主函数：演示所有概念
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "C++ 引用详解 - 推理加速基础\n";
    std::cout << "========================================\n";
    
    // 1. 基础交换示例
    std::cout << "\n--- 1. 基础交换示例 ---\n";
    int a = 10, b = 20;
    std::cout << "交换前: a=" << a << ", b=" << b << "\n";
    
    // C风格（需要取地址）
    swap_c_style(&a, &b);
    std::cout << "C风格交换后: a=" << a << ", b=" << b << "\n";
    
    // C++风格（直接传变量）
    swap_cpp_style(a, b);
    std::cout << "C++风格交换后: a=" << a << ", b=" << b << "\n";
    
    // 2. Tensor传递示例
    std::cout << "\n--- 2. Tensor传递对比 ---\n";
    std::cout << "创建Tensor (模拟1MB数据):\n";
    Tensor tensor("hidden_states", {1, 256, 1024});  // 约1MB
    
    std::cout << "\n按值传递 (会拷贝!):\n";
    process_tensor_by_value(tensor);  // 触发拷贝构造
    
    std::cout << "\n按const引用传递 (零拷贝):\n";
    process_tensor_by_ref(tensor);    // 无拷贝
    
    // 3. 原地操作示例
    std::cout << "\n--- 3. 原地操作 (in-place) ---\n";
    tensor.data[0] = -1.0f;  // 设置一个负值
    std::cout << "ReLU前 data[0] = " << tensor.data[0] << "\n";
    relu_inplace(tensor);     // 使用非const引用修改
    std::cout << "ReLU后 data[0] = " << tensor.data[0] << "\n";
    
    // 4. KV Cache示例
    std::cout << "\n--- 4. KV Cache 引用返回 ---\n";
    KVCache cache;
    cache.add("key_cache", {1, 32, 128, 64});   // 模拟Key缓存
    cache.add("value_cache", {1, 32, 128, 64}); // 模拟Value缓存
    
    // 通过引用直接访问缓存项
    Tensor& key_ref = cache.get(0);
    std::cout << "直接访问缓存: " << key_ref.name << "\n";
    
    // 5. 引用 vs 指针对比
    demo_reference_vs_pointer();
    
    // 6. Transformer层示例
    std::cout << "\n--- 6. Transformer层前向传播 ---\n";
    Tensor input("input", {1, 128, 768});
    Tensor weights("attention_weights", {768, 768});
    Tensor output("output", {1, 128, 768});
    
    transformer_layer_forward(input, weights, output);
    
    std::cout << "\n========================================\n";
    std::cout << "总结：在推理加速中，合理使用引用可以实现零拷贝，\n";
    std::cout << "对于GB级别的Tensor数据，这是关键的性能优化手段。\n";
    std::cout << "========================================\n";
    
    return 0;
}

// ============================================================================
// 练习题
// ============================================================================
/*
【练习1】
实现一个函数 void scale_tensor(Tensor& t, float factor)
将Tensor中的所有元素乘以factor，使用引用实现原地操作。
void scale_tensor(Tensor& t, float factor) {
    for (float& val : t.data) {
        val *= factor;
    }
}
【练习2】
实现一个函数 float compute_mean(const Tensor& t)
计算Tensor所有元素的平均值，使用const引用保证不修改原数据。
float compute_mean(const Tensor& t) {
    float sum = 0.0f;
    for (float val : t.data) {
        sum += val;
    }
    return sum / t.data.size();
}
【练习3】
思考：为什么Tensor的拷贝构造函数中要输出警告信息？
在实际推理框架中，意外的Tensor拷贝会导致什么问题？
造成莫名的性能原因，因为拷贝构造函数会触发数据的深拷贝，
【练习4】
修改KVCache类，添加一个方法：
bool has(const std::string& name) const
返回是否存在指定名称的缓存项。
*/
class KVCache {
private:
    std::vector<Tensor> cache;
    
public:
    // 添加新的缓存项
    void add(const std::string& name, std::vector<int> shape) {
        cache.emplace_back(name, shape);
    }
    // 查找是否存在指定名称的缓存项
    bool has(const std::string& name) const {
        for (const auto& tensor : cache) {
            if (tensor.name == name) {
                return true;
            }
        }
        return false;
    }
    /**
     * 返回引用：允许直接访问和修改缓存中的Tensor
     * 注意：返回引用时要确保对象的生命周期足够长
     */
    Tensor& get(size_t index) {
        if (index >= cache.size()) {
            throw std::out_of_range("缓存索引越界");
        }
        return cache[index];  // 返回引用，不拷贝
    }
    
    // 返回const引用：只读访问
    const Tensor& get_readonly(size_t index) const {
        return cache[index];
    }
    
    size_t size() const { return cache.size(); }
};
