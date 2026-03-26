# 大模型推理加速学习代码库

> 🚀 从C语言基础出发，系统学习C++和Python在大模型推理加速中的应用

## 📖 项目简介

本代码库专为有C语言基础、希望深入学习大模型推理加速的开发者设计。通过循序渐进的代码示例和详尽的中文注释，帮助您掌握现代C++和Python的高级特性，并理解它们在实际推理加速场景中的应用。

## 🎯 适用人群

- ✅ 掌握C语言基础
- ✅ 了解Python基本语法
- ✅ 对C++有初步了解
- ✅ 希望学习大模型推理加速技术

## 📚 目录结构

```
code_learn/
├── 01_cpp_basics/               # C++ 基础（从C过渡）
│   ├── 01_from_c_to_cpp/        # C到C++的过渡
│   ├── 02_classes_objects/      # 类与对象
│   ├── 03_templates/            # 模板编程
│   └── 04_stl/                  # 标准模板库
├── 02_cpp_advanced/             # C++ 进阶
│   ├── 01_smart_pointers/       # 智能指针
│   ├── 02_move_semantics/       # 移动语义
│   ├── 03_multithreading/       # 多线程编程
│   └── 04_simd/                 # SIMD向量化
├── 03_python_advanced/          # Python 进阶
│   ├── 01_decorators/           # 装饰器
│   ├── 02_generators/           # 生成器
│   ├── 03_numpy_optimization/   # NumPy优化
│   └── 04_multiprocessing/      # 多进程
├── 04_inference_acceleration/   # 推理加速核心技术
│   ├── 01_memory_optimization/  # 内存优化
│   ├── 02_quantization/         # 量化技术
│   ├── 03_attention_optimization/ # 注意力优化
│   └── 04_parallel_strategies/  # 并行策略
└── 05_projects/                 # 实战项目
    ├── 01_simple_inference/     # 简单推理引擎
    ├── 02_transformer_opt/      # Transformer优化
    └── 03_benchmarking/         # 性能基准测试
```

## 🛠️ 环境要求

### C++ 编译器
- GCC 9+ 或 Clang 10+ 或 MSVC 2019+
- 支持 C++17 标准

### Python 环境
- Python 3.8+
- NumPy >= 1.20

### 可选依赖
- PyTorch（用于深度学习相关示例）
- OpenMP（用于并行计算示例）

## 🚀 快速开始

### 编译 C++ 示例

```bash
# Linux/macOS
g++ -std=c++17 -O2 -o example example.cpp
./example

# Windows (MSVC)
cl /std:c++17 /O2 example.cpp
example.exe
```

### 运行 Python 示例

```bash
python example.py
```

## 📝 学习路径

### 阶段一：C++ 基础（建议2周）
从C语言平滑过渡到C++，掌握类、模板、STL等核心概念。

### 阶段二：C++ 进阶（建议2周）
深入智能指针、移动语义、多线程、SIMD等高级特性。

### 阶段三：Python 进阶（建议1周）
掌握装饰器、生成器、NumPy优化等提升Python性能的技巧。

### 阶段四：推理加速核心（建议3周）
学习KV Cache、量化、FlashAttention、并行策略等核心技术。

### 阶段五：实战项目（建议2周）
动手实现简单推理引擎，综合应用所学知识。

## 💡 代码特点

1. **详尽的中文注释** - 每行关键代码都有解释
2. **对比学习** - C vs C++、朴素实现 vs 优化实现
3. **实际场景** - 每个知识点都链接到推理加速应用
4. **可运行示例** - 所有代码都可以独立编译运行

## 📖 推荐阅读

- 《C++ Primer》- C++入门经典
- 《Effective Modern C++》- 现代C++最佳实践
- 《Python高性能编程》- Python性能优化
- vLLM、TensorRT-LLM 源码 - 工业级推理框架

---

**开始学习吧！** 建议从 `01_cpp_basics/01_from_c_to_cpp/` 开始。
