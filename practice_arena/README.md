# Interview Practice Arena

为 `code_learn` 题库准备的网页练习台，目标是把“选题 -> 写代码 -> 跑 correctness -> 保存草稿”这条链路压缩到一个浏览器页面里。

## 功能

- 题目自动发现：扫描 `pytorch_basics`、`model_layers`、`triton`、`cuda` 下的练习目录。
- 真编辑器：CodeMirror 6（行号、Python / CUDA 语法高亮、括号匹配、Tab 缩进、Ctrl+Z 历史），已 vendored 到 `static/vendor/`，**浏览器不需要联网**，加载失败时自动退回 textarea。
- 两种模式：优先支持 `exercise.*` 的自动测试，也支持 `interview.*` 的随机面试练习。
- 草稿隔离：网页里编辑的代码默认保存到 `.practice_arena/drafts/`，不覆盖原题库模板。
- 训练记录：保存做题状态、累计时长、最近执行历史、最佳分数。
- 面试计时：支持开始、暂停并记账，时长持久化到 `.practice_arena/state/arena_state.json`。
- 答案面板：支持查看参考答案、对比当前代码和参考答案的 unified diff。
- 提交评分：按自动测试/可运行性/占位符清理做启发式评分，适合面试自测，不是严格裁判。
- 一键执行：
  - `exercise.*` + `test.py` 时，直接跑 `pytest`
  - `interview.*` / 没有测试的题时，直接运行当前文件
- 随机面试题：从支持 `interview.*` 的题目中随机抽题。

## 运行

### 在 187（或任何带 GPU 的远程机）启动服务

```bash
ssh 187
source /home/yangfu/anaconda3/etc/profile.d/conda.sh
conda activate vllm-env
cd /home/yangfu/workspace/code_learn
python -m pip install -r practice_arena/requirements.txt
uvicorn practice_arena.app:app --host 127.0.0.1 --port 8765 --reload
```

### 推荐：本地浏览器走 SSH 端口转发访问（无需公网暴露）

在自己笔记本上另开一个终端：

```bash
ssh -N -L 8765:localhost:8765 187
```

保留这个连接，浏览器打开 `http://127.0.0.1:8765` 即可。这样：

- 服务只绑 127.0.0.1，不会被局域网其他人扫到
- 无需配置 nginx / 反向代理 / SSL
- 关掉 SSH 隧道连接就关了，对外完全不可见
- **CodeMirror 编辑器走的是本地 `static/vendor/`**，浏览器和服务器之间不需要任何额外的公网请求

### 想直接局域网访问（不推荐放公网）

把 `--host 127.0.0.1` 改成 `--host 0.0.0.0`，然后访问 `http://<远程机内网IP>:8765`。注意先用防火墙限制访问来源。

## 重新拉取 CodeMirror 资源

CodeMirror 6 的依赖图已经通过 [`vendor.py`](vendor.py) 提前拉到 `static/vendor/` 并随仓库一起提交，正常情况下不用碰。

只有以下场景才需要重新跑 vendor：

- 想升级 CodeMirror 到新版本 → 改 `vendor.py` 里 `ENTRY_POINTS` 的版本号
- 删了 `static/vendor/` 想重建
- 浏览器报某个模块 404，说明 BFS 漏抓了

```bash
# 服务器需要能访问 https://esm.sh，本地浏览器不需要
uv run python -m practice_arena.vendor
```

脚本是幂等的，每次跑都会清空 `static/vendor/` 后重抓一次。

## 版本管理建议

- 练习台源码走 Git 分支，例如当前开发分支：`feature/practice-arena`
- 个人练习草稿不进版本库，统一落在 `.practice_arena/`
- 如果后续想加排行榜、题目标记、做题记录，再单独把元数据放进一个可控的 JSON/SQLite 层，不要直接污染题库源码
