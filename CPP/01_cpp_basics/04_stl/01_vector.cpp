/**
 * @file 01_vector.cpp
 * @brief std::vector - 动态数组
 * 
 * 【推理场景】
 * - Tensor数据存储
 * - 形状/步长信息
 * - 批量处理
 * 
 * 编译: g++ -std=c++17 -o 01_vector 01_vector.cpp
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

int main() {
    std::cout << "========================================\n";
    std::cout << "std::vector\n";
    std::cout << "========================================\n";
    
    // 1. 创建与初始化
    std::cout << "\n--- 创建 ---\n";
    std::vector<float> v1;                    // 空vector
    std::vector<float> v2(10);                // 10个0
    std::vector<float> v3(10, 1.0f);          // 10个1.0
    std::vector<float> v4 = {1, 2, 3, 4, 5};  // 初始化列表
    
    std::cout << "v3.size() = " << v3.size() << "\n";
    std::cout << "v4 = [" << v4[0] << ", " << v4[1] << ", ...]\n";
    
    // 2. 添加元素
    std::cout << "\n--- 添加元素 ---\n";
    v1.push_back(1.0f);
    v1.push_back(2.0f);
    v1.emplace_back(3.0f);  // 更高效，直接构造
    std::cout << "v1.size() = " << v1.size() << "\n";
    
    // 3. 访问元素
    std::cout << "\n--- 访问 ---\n";
    std::cout << "v4[0] = " << v4[0] << "\n";
    std::cout << "v4.at(1) = " << v4.at(1) << " (带边界检查)\n";
    std::cout << "v4.front() = " << v4.front() << "\n";
    std::cout << "v4.back() = " << v4.back() << "\n";
    std::cout << "v4.data() = " << v4.data() << " (原始指针)\n";
    
    // 4. 容量管理
    std::cout << "\n--- 容量 ---\n";
    std::vector<float> weights;
    weights.reserve(1000);  // 预分配，避免多次扩容
    std::cout << "capacity = " << weights.capacity() << "\n";
    std::cout << "size = " << weights.size() << "\n";
    
    // 5. 遍历
    std::cout << "\n--- 遍历 ---\n";
    std::cout << "范围for: ";
    for (float val : v4) {
        std::cout << val << " ";
    }
    std::cout << "\n";
    
    // 6. 与算法配合
    std::cout << "\n--- 算法 ---\n";
    float sum = std::accumulate(v4.begin(), v4.end(), 0.0f);
    std::cout << "sum = " << sum << "\n";
    
    auto max_it = std::max_element(v4.begin(), v4.end());
    std::cout << "max = " << *max_it << " at index " << (max_it - v4.begin()) << "\n";
    
    // 7. 2D数据（Tensor模拟）
    std::cout << "\n--- 2D数据 ---\n";
    int rows = 3, cols = 4;
    std::vector<float> matrix(rows * cols);
    
    // 行优先访问
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = i * cols + j;
        }
    }
    
    std::cout << "matrix[1][2] = " << matrix[1 * cols + 2] << "\n";
    
    // 8. 移动语义（避免拷贝）
    std::cout << "\n--- 移动语义 ---\n";
    std::vector<float> large(1000000);
    std::cout << "移动前 large.size() = " << large.size() << "\n";
    
    std::vector<float> moved = std::move(large);
    std::cout << "移动后 large.size() = " << large.size() << "\n";
    std::cout << "moved.size() = " << moved.size() << "\n";
    
    return 0;
}
