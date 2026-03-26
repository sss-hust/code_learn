/**
 * @file 02_const_constexpr.cpp
 * @brief const 和 constexpr 详解 - 编译期优化的基础
 * 
 * 【推理加速场景】
 * 在大模型推理中，const和constexpr用于：
 * 1. 定义模型配置常量（hidden_size, num_heads等）
 * 2. 编译期计算维度信息，减少运行时开销
 * 3. 帮助编译器进行优化（常量折叠、内联等）
 * 4. 保护不应修改的数据（模型权重）
 * 
 * 编译: g++ -std=c++17 -O2 -o 02_const_constexpr 02_const_constexpr.cpp
 * 运行: ./02_const_constexpr
 */

#include <iostream>
#include <array>
#include <cmath>

// ============================================================================
// 第一部分：const 基础
// ============================================================================

/**
 * 【C语言回顾】
 * C语言中的const有一些限制：
 * - 不能用于数组大小（某些编译器支持VLA除外）
 * - 编译器不一定将其视为编译期常量
 */

// C++中的const变量是真正的常量
const int MAX_BATCH_SIZE = 32;           // 最大批次大小
const int MAX_SEQ_LENGTH = 2048;         // 最大序列长度
const float EPSILON = 1e-6f;             // 数值稳定性常量

// 可以用const定义数组大小（C++特性）
const int HIDDEN_DIM = 768;
float weights[HIDDEN_DIM];  // C中不行，C++可以

/**
 * 【const与指针】经典难题
 * 口诀：const在*左边修饰数据，在*右边修饰指针
 */
void demo_const_pointer() {
    std::cout << "\n=== const与指针 ===\n";
    
    int value = 10;
    int other = 20;
    
    // 情况1：指向常量的指针（底层const）
    const int* ptr1 = &value;
    // *ptr1 = 20;           // 错误！不能通过ptr1修改
    ptr1 = &other;           // 可以，指针本身可以改变
    
    // 情况2：常量指针（顶层const）
    int* const ptr2 = &value;
    *ptr2 = 30;              // 可以，可以修改指向的值
    // ptr2 = &other;        // 错误！指针本身不能改变
    
    // 情况3：指向常量的常量指针
    const int* const ptr3 = &value;
    // *ptr3 = 40;           // 错误！
    // ptr3 = &other;        // 错误！
    
    std::cout << "const int* : 数据不可变，指针可变\n";
    std::cout << "int* const : 数据可变，指针不可变\n";
    std::cout << "const int* const : 都不可变\n";
}

// ============================================================================
// 第二部分：const 成员函数
// ============================================================================

/**
 * 【场景】模型配置类
 * const成员函数保证不会修改对象状态，可以被const对象调用
 */
class ModelConfig {
private:
    int hidden_size_;
    int num_heads_;
    int num_layers_;
    float dropout_rate_;
    
public:
    ModelConfig(int hidden, int heads, int layers, float dropout = 0.0f)
        : hidden_size_(hidden), num_heads_(heads), 
          num_layers_(layers), dropout_rate_(dropout) {}
    
    // const成员函数：只读访问器
    int hidden_size() const { return hidden_size_; }
    int num_heads() const { return num_heads_; }
    int num_layers() const { return num_layers_; }
    float dropout_rate() const { return dropout_rate_; }
    
    // 计算每个头的维度 - const函数可以进行计算
    int head_dim() const { 
        return hidden_size_ / num_heads_; 
    }
    
    // 计算总参数量（近似）
    long long total_params() const {
        // 简化计算：每层约12*hidden^2参数
        return static_cast<long long>(num_layers_) * 12 * 
               hidden_size_ * hidden_size_;
    }
    
    // 打印配置 - const函数
    void print() const {
        std::cout << "ModelConfig:\n";
        std::cout << "  hidden_size: " << hidden_size_ << "\n";
        std::cout << "  num_heads: " << num_heads_ << "\n";
        std::cout << "  head_dim: " << head_dim() << "\n";
        std::cout << "  num_layers: " << num_layers_ << "\n";
        std::cout << "  total_params: ~" << total_params() / 1e6 << "M\n";
    }
    
    // 非const函数：修改器
    void set_dropout(float rate) {
        dropout_rate_ = rate;  // 可以修改成员
    }
};

/**
 * 使用const引用传递配置
 * 保证配置不会被意外修改
 */
void initialize_model(const ModelConfig& config) {
    std::cout << "\n初始化模型...\n";
    config.print();  // 可以调用const成员函数
    // config.set_dropout(0.1);  // 错误！不能调用非const函数
}

// ============================================================================
// 第三部分：constexpr - 编译期计算
// ============================================================================

/**
 * 【关键区别】
 * const：运行时常量（值不可变，但可能在运行时确定）
 * constexpr：编译期常量（值必须在编译时确定）
 */

// constexpr变量：编译期确定的常量
constexpr int NUM_ATTENTION_HEADS = 12;
constexpr int HIDDEN_SIZE = 768;
constexpr int HEAD_DIM = HIDDEN_SIZE / NUM_ATTENTION_HEADS;  // 编译期计算

// constexpr函数：可以在编译期执行
constexpr int calculate_tensor_size(int batch, int seq_len, int hidden) {
    return batch * seq_len * hidden;
}

// 编译期计算内存需求
constexpr size_t ACTIVATION_MEMORY = 
    calculate_tensor_size(32, 2048, 768) * sizeof(float);

/**
 * 更复杂的constexpr函数示例
 * C++14起，constexpr函数可以包含更复杂的逻辑
 */
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

constexpr double power(double base, int exp) {
    double result = 1.0;
    for (int i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

// 编译期生成查找表
constexpr std::array<int, 10> generate_factorials() {
    std::array<int, 10> table{};
    for (int i = 0; i < 10; ++i) {
        table[i] = factorial(i);
    }
    return table;
}

// 编译期生成的阶乘表 - 运行时零开销！
constexpr auto FACTORIAL_TABLE = generate_factorials();

// ============================================================================
// 第四部分：推理优化中的实际应用
// ============================================================================

/**
 * 【场景】编译期确定Attention维度
 * 在模板元编程中非常有用
 */
template<int HiddenSize, int NumHeads>
class AttentionConfig {
public:
    static constexpr int hidden_size = HiddenSize;
    static constexpr int num_heads = NumHeads;
    static constexpr int head_dim = HiddenSize / NumHeads;
    
    // 编译期检查
    static_assert(HiddenSize % NumHeads == 0, 
                  "hidden_size must be divisible by num_heads");
    
    // 编译期计算QKV总大小
    static constexpr int qkv_size = 3 * HiddenSize;
    
    // 可以用于数组大小
    float query_buffer[head_dim];  // 编译期确定大小
    
    void print_config() const {
        std::cout << "AttentionConfig<" << HiddenSize << ", " << NumHeads << ">:\n";
        std::cout << "  head_dim: " << head_dim << "\n";
        std::cout << "  qkv_size: " << qkv_size << "\n";
    }
};

/**
 * 【场景】编译期生成位置编码表
 * 避免运行时计算三角函数
 */
template<int MaxLen, int Dim>
class PositionalEncoding {
private:
    // 编译期不能用std::sin，这里用运行时初始化演示思路
    float encoding[MaxLen][Dim];
    
public:
    PositionalEncoding() {
        // 实际推理框架中，这个表会预先计算好
        for (int pos = 0; pos < MaxLen; ++pos) {
            for (int i = 0; i < Dim; ++i) {
                float angle = pos / std::pow(10000.0f, (2.0f * (i / 2)) / Dim);
                encoding[pos][i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
            }
        }
    }
    
    const float* get(int pos) const {
        return encoding[pos];
    }
};

// ============================================================================
// 第五部分：if constexpr - 编译期分支
// ============================================================================

/**
 * 【C++17特性】if constexpr
 * 在编译期决定执行哪个分支，未选中的分支不会被编译
 * 非常适合模板中的类型判断
 */
template<typename T>
void process_data(const T* data, int size) {
    if constexpr (std::is_same_v<T, float>) {
        std::cout << "处理float数据，可以使用AVX指令\n";
        // 这里可以使用float特定的SIMD优化
    } else if constexpr (std::is_same_v<T, int>) {
        std::cout << "处理int数据（量化后）\n";
        // int8量化后的处理逻辑
    } else {
        std::cout << "处理其他类型数据\n";
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "const 和 constexpr 详解\n";
    std::cout << "========================================\n";
    
    // 1. const基础
    std::cout << "\n--- 1. const基础常量 ---\n";
    std::cout << "MAX_BATCH_SIZE = " << MAX_BATCH_SIZE << "\n";
    std::cout << "MAX_SEQ_LENGTH = " << MAX_SEQ_LENGTH << "\n";
    std::cout << "EPSILON = " << EPSILON << "\n";
    
    // 2. const与指针
    demo_const_pointer();
    
    // 3. const成员函数
    std::cout << "\n--- 3. const成员函数 ---\n";
    ModelConfig gpt2_config(768, 12, 12);
    initialize_model(gpt2_config);
    
    const ModelConfig llama_config(4096, 32, 32);
    llama_config.print();  // const对象只能调用const函数
    // llama_config.set_dropout(0.1);  // 错误！
    
    // 4. constexpr编译期计算
    std::cout << "\n--- 4. constexpr编译期计算 ---\n";
    std::cout << "HEAD_DIM = " << HEAD_DIM << " (编译期计算)\n";
    std::cout << "ACTIVATION_MEMORY = " << ACTIVATION_MEMORY / 1024.0 / 1024.0 
              << " MB (编译期计算)\n";
    
    // 编译期阶乘表
    std::cout << "\n编译期生成的阶乘表:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "  " << i << "! = " << FACTORIAL_TABLE[i] << "\n";
    }
    
    // 5. 模板中的constexpr
    std::cout << "\n--- 5. 模板中的constexpr ---\n";
    AttentionConfig<768, 12> bert_attention;
    bert_attention.print_config();
    
    AttentionConfig<4096, 32> llama_attention;
    llama_attention.print_config();
    
    // 6. if constexpr
    std::cout << "\n--- 6. if constexpr ---\n";
    float float_data[10];
    int int_data[10];
    process_data(float_data, 10);
    process_data(int_data, 10);
    
    std::cout << "\n========================================\n";
    std::cout << "总结：\n";
    std::cout << "- const: 运行时常量，保护数据不被修改\n";
    std::cout << "- constexpr: 编译期常量，零运行时开销\n";
    std::cout << "- 在推理优化中，用constexpr预计算维度、生成查找表\n";
    std::cout << "========================================\n";
    
    return 0;
}

// ============================================================================
// 练习题
// ============================================================================
/*
【练习1】
编写一个constexpr函数 compute_memory_bytes(int batch, int seq, int hidden, int dtype_size)
计算Tensor所需的内存字节数。
constexpr int compute_memory_bytes(int batch, int seq, int hidden, int dtype_size) {
    return batch * seq * hidden * dtype_size;
}
【练习2】
使用constexpr生成一个RoPE（旋转位置编码）中使用的频率表。
提示：freq = 1.0 / pow(10000, 2*i/dim)
constexpr float frequency_table[] = {
    1.0 / pow(10000, 2*0/768),
    1.0 / pow(10000, 2*1/768),
    ...
};
【练习3】
为ModelConfig类添加一个const成员函数 memory_estimate()
返回模型大致需要的内存量（MB）。
float memory_estimate() const {
    return static_cast<float>(total_params()) / 1e6;
}
【练习4】
思考：为什么推理框架中大量使用constexpr？
相比运行时计算，编译期计算有什么优势？
运行前就确定好了变量的值，同时有分支之类的不需要编译，节省时间
*/ 
