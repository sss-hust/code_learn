# code_learn

> 一个面向"PyTorch / Triton / CUDA 推理加速"的渐进练习题库 + 浏览器练习台。

## 这是什么

按"梯度"组织的 4 个题库目录，配一个 FastAPI 写的网页练习台。从张量基本操作一路练到 Flash Attention：

| 目录 | 内容 | 题量 |
|---|---|---|
| [`pytorch_basics/`](pytorch_basics/) | PyTorch 基础（tensor / 广播 / autograd / 训练循环 / 手写 Linear） | 10 |
| [`model_layers/`](model_layers/) | `nn.Module` 风格的模型层（Embedding / RMSNorm / MHA / GQA / MoE / TransformerBlock） | 11 |
| [`triton/`](triton/) | Triton kernel（向量加法 / 行 reduction / softmax / RoPE / matmul / Flash Attention） | 14 |
| [`cuda/`](cuda/) | CUDA kernel（vector_add / 2D 索引 / warp shuffle / shared memory / GEMM / Flash Attention） | 14 |
| [`practice_arena/`](practice_arena/) | 浏览器在线练习台（CodeMirror 编辑器 + 自动测试 + 计时 + 评分） | webapp |

每个题目有三种文件：

- `interview.py` / `interview.cu`：空骨架，自己填
- `solution.py` / `solution.cu`：参考答案
- `test.py`：pytest 自动校验（默认测 interview 实现，加 `--check-solution` 改测参考答案）

## 推荐学习路径

```
pytorch_basics  →  model_layers  →  triton / cuda  →  practice_arena 刷题
   (基础)         (会写 nn.Module)    (写 GPU kernel)        (刷面试 / 计时)
```

每个分类的 README 里有更细的推荐顺序：[pytorch_basics/README.md](pytorch_basics/README.md)、[triton/README.md](triton/README.md)、[cuda/README.md](cuda/README.md)、[model_layers/README.md](model_layers/README.md)。

## 用 uv 快速跑起来

仓库用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境，一条命令装齐所有依赖：

```bash
# 1. 安装 uv（如果还没装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 进入仓库并同步依赖（自动建 .venv/ + 装 torch / triton / fastapi 等）
cd code_learn
uv sync

# 3. 验证：随便跑一道 pytorch_basics 的参考答案
uv run pytest pytorch_basics/01_tensor_basics/test.py --check-solution -v
```

> uv 第一次 sync 会下载约 2-3 GB 的 PyTorch + CUDA wheel（Linux x86_64 默认带 CUDA 12.x runtime，需要主机有兼容驱动）。Mac / Windows 上没有 CUDA 时 triton 会自动跳过（pyproject.toml 里有 platform marker）。

## 跑某道题

```bash
# Python / Triton 题：直接跑 interview / solution
uv run python pytorch_basics/01_tensor_basics/solution.py
uv run python triton/01_vector_add/solution.py

# CUDA 题：先 nvcc 编译再运行
cd cuda/01_vector_add
nvcc -O2 -o solution solution.cu && ./solution

# 自动测试某道题（默认测 interview 实现）
cd pytorch_basics/01_tensor_basics
uv run pytest test.py -v

# 测参考答案
uv run pytest test.py -v --check-solution
```

## 启动浏览器练习台

```bash
# 在带 GPU 的服务器上启动（绑 127.0.0.1，避免暴露公网）
uv run uvicorn practice_arena.app:app --host 127.0.0.1 --port 8765 --reload
```

本地通过 SSH 端口转发访问：

```bash
# 笔记本另开一个终端
ssh -N -L 8765:localhost:8765 <你的服务器>
# 浏览器打开 http://127.0.0.1:8765
```

详见 [practice_arena/README.md](practice_arena/README.md)。

## 仓库结构

```
code_learn/
├── pyproject.toml        # uv 项目配置 + 依赖清单
├── .python-version       # 钉死 Python 3.13
├── conftest.py           # 全局 --check-solution flag
├── pytorch_basics/       # 10 道 PyTorch 基础题
├── model_layers/         # 11 道 nn.Module 模型层题
├── triton/               # 14 道 Triton kernel 题
├── cuda/                 # 14 道 CUDA kernel 题
├── practice_arena/       # FastAPI 浏览器练习台
└── CPP/                  # （早期 C++ 学习材料，与主线无关）
```

## 不用 uv 的替代方式

如果机器上已经有 conda env / pip 装好了 torch + triton，也可以直接：

```bash
pip install -r practice_arena/requirements.txt
python -m pytest pytorch_basics/01_tensor_basics/test.py --check-solution
```

uv.lock 是用来锁定可复现版本组合的，迁移到新机器的最快方式是 `uv sync --frozen`。
