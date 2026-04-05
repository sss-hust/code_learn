# Interview Practice Arena

为 `code_learn` 题库准备的网页练习台，目标是把“选题 -> 写代码 -> 跑 correctness -> 保存草稿”这条链路压缩到一个浏览器页面里。

## 功能

- 题目自动发现：扫描 `model_layers`、`triton`、`cuda` 下的练习目录。
- 两种模式：优先支持 `exercise.*` 的自动测试，也支持 `interview.*` 的随机面试练习。
- 草稿隔离：网页里编辑的代码默认保存到 `.practice_arena/drafts/`，不覆盖原题库模板。
- 一键执行：
  - `exercise.*` + `test.py` 时，直接跑 `pytest`
  - `interview.*` / 没有测试的题时，直接运行当前文件
- 随机面试题：从支持 `interview.*` 的题目中随机抽题。

## 运行

```bash
ssh 187
source /home/yangfu/anaconda3/etc/profile.d/conda.sh
conda activate vllm-env
cd /home/yangfu/workspace/code_learn
python -m pip install -r practice_arena/requirements.txt
uvicorn practice_arena.app:app --host 0.0.0.0 --port 8765 --reload
```

浏览器访问 `http://<187的IP>:8765`。

## 版本管理建议

- 练习台源码走 Git 分支，例如当前开发分支：`feature/practice-arena`
- 个人练习草稿不进版本库，统一落在 `.practice_arena/`
- 如果后续想加排行榜、题目标记、做题记录，再单独把元数据放进一个可控的 JSON/SQLite 层，不要直接污染题库源码
