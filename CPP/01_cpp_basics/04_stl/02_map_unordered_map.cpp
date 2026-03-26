/**
 * @file 02_map_unordered_map.cpp
 * @brief 关联容器 - 键值对存储
 * 
 * 【推理场景】
 * - 模型权重命名查找
 * - KV Cache管理
 * - 配置参数
 * 
 * 编译: g++ -std=c++17 -o 02_map 02_map_unordered_map.cpp
 */

#include <iostream>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>

int main() {
    std::cout << "========================================\n";
    std::cout << "关联容器\n";
    std::cout << "========================================\n";
    
    // ========== std::map (有序，红黑树) ==========
    std::cout << "\n--- std::map (有序) ---\n";
    
    std::map<std::string, int> layer_sizes;
    layer_sizes["embedding"] = 50257 * 768;
    layer_sizes["attention"] = 768 * 768 * 4;
    layer_sizes["ffn"] = 768 * 3072 * 2;
    
    // 插入
    layer_sizes.insert({"output", 768 * 50257});
    layer_sizes["lm_head"] = 768 * 50257;
    
    // 遍历（按key排序）
    std::cout << "层参数量:\n";
    for (const auto& [name, size] : layer_sizes) {
        std::cout << "  " << name << ": " << size / 1e6 << "M\n";
    }
    
    // 查找
    if (layer_sizes.count("attention")) {
        std::cout << "attention存在\n";
    }
    
    auto it = layer_sizes.find("ffn");
    if (it != layer_sizes.end()) {
        std::cout << "ffn: " << it->second << "\n";
    }
    
    // ========== std::unordered_map (无序，哈希表) ==========
    std::cout << "\n--- std::unordered_map (哈希) ---\n";
    
    // 模拟KV Cache
    std::unordered_map<int, std::vector<float>> kv_cache;
    
    for (int layer = 0; layer < 3; ++layer) {
        kv_cache[layer] = std::vector<float>(128 * 64, 0.0f);
    }
    
    std::cout << "KV Cache层数: " << kv_cache.size() << "\n";
    std::cout << "Layer 0 cache大小: " << kv_cache[0].size() << "\n";
    
    // 性能比较
    std::cout << "\n--- 性能特点 ---\n";
    std::cout << "map:\n";
    std::cout << "  - 查找: O(log n)\n";
    std::cout << "  - 有序遍历\n";
    std::cout << "  - 适合：需要顺序的场景\n";
    
    std::cout << "unordered_map:\n";
    std::cout << "  - 查找: O(1) 平均\n";
    std::cout << "  - 无序\n";
    std::cout << "  - 适合：频繁查找的场景（如权重表）\n";
    
    // 实际场景：模型权重表
    std::cout << "\n--- 模型权重表 ---\n";
    std::unordered_map<std::string, float*> weight_table;
    
    float w1[10], w2[20], w3[30];  // 模拟权重
    weight_table["layer.0.weight"] = w1;
    weight_table["layer.0.bias"] = w2;
    weight_table["layer.1.weight"] = w3;
    
    // 快速查找
    if (auto it = weight_table.find("layer.0.weight"); it != weight_table.end()) {
        std::cout << "找到权重: " << it->first << "\n";
    }
    
    return 0;
}
