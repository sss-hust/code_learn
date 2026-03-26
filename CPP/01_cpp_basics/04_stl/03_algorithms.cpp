/**
 * @file 03_algorithms.cpp
 * @brief STL算法 - 高效数据处理
 * 
 * 【推理场景】
 * - 数据预处理/后处理
 * - 查找最大值（argmax）
 * - 排序（top-k采样）
 * 
 * 编译: g++ -std=c++17 -o 03_algorithms 03_algorithms.cpp
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

int main() {
    std::cout << "========================================\n";
    std::cout << "STL算法\n";
    std::cout << "========================================\n";
    
    std::vector<float> logits = {2.1f, 3.5f, 1.2f, 4.8f, 0.5f, 3.2f};
    
    // ========== 1. 查找 ==========
    std::cout << "\n--- 查找 ---\n";
    
    // 最大值（argmax）
    auto max_it = std::max_element(logits.begin(), logits.end());
    int argmax = max_it - logits.begin();
    std::cout << "argmax = " << argmax << ", value = " << *max_it << "\n";
    
    // 最小值
    auto min_it = std::min_element(logits.begin(), logits.end());
    std::cout << "min = " << *min_it << "\n";
    
    // 查找特定值
    auto found = std::find(logits.begin(), logits.end(), 3.5f);
    if (found != logits.end()) {
        std::cout << "找到3.5，位置: " << (found - logits.begin()) << "\n";
    }
    
    // ========== 2. 变换 ==========
    std::cout << "\n--- 变换 ---\n";
    
    // 就地变换（如ReLU）
    std::vector<float> data = {-1.0f, 2.0f, -3.0f, 4.0f};
    std::transform(data.begin(), data.end(), data.begin(),
        [](float x) { return x > 0 ? x : 0.0f; });
    
    std::cout << "ReLU: [";
    for (float v : data) std::cout << v << " ";
    std::cout << "]\n";
    
    // ========== 3. 累积 ==========
    std::cout << "\n--- 累积 ---\n";
    
    float sum = std::accumulate(logits.begin(), logits.end(), 0.0f);
    std::cout << "sum = " << sum << "\n";
    
    float product = std::accumulate(logits.begin(), logits.end(), 1.0f,
        std::multiplies<float>());
    std::cout << "product = " << product << "\n";
    
    // ========== 4. 排序 ==========
    std::cout << "\n--- 排序 ---\n";
    
    // Top-K：部分排序
    std::vector<float> probs = logits;
    int k = 3;
    std::partial_sort(probs.begin(), probs.begin() + k, probs.end(),
        std::greater<float>());  // 降序
    
    std::cout << "Top-" << k << ": [";
    for (int i = 0; i < k; ++i) std::cout << probs[i] << " ";
    std::cout << "]\n";
    
    // 排序索引（argsort）
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&logits](int a, int b) { return logits[a] > logits[b]; });
    
    std::cout << "argsort (降序): [";
    for (int i : indices) std::cout << i << " ";
    std::cout << "]\n";
    
    // ========== 5. 填充与生成 ==========
    std::cout << "\n--- 填充 ---\n";
    
    std::vector<float> zeros(5);
    std::fill(zeros.begin(), zeros.end(), 0.0f);
    
    std::vector<int> seq(10);
    std::iota(seq.begin(), seq.end(), 0);  // 0,1,2,...,9
    std::cout << "序列: [";
    for (int v : seq) std::cout << v << " ";
    std::cout << "]\n";
    
    // ========== 6. 条件判断 ==========
    std::cout << "\n--- 条件 ---\n";
    
    bool all_pos = std::all_of(logits.begin(), logits.end(),
        [](float x) { return x > 0; });
    std::cout << "全部正数: " << all_pos << "\n";
    
    int count_big = std::count_if(logits.begin(), logits.end(),
        [](float x) { return x > 2.0f; });
    std::cout << "大于2的数量: " << count_big << "\n";
    
    return 0;
}
