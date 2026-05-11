"""Vendor CodeMirror 6 ESM 资源到 ``static/vendor/``，让浏览器不再依赖外网 CDN。

esm.sh 把 ``codemirror`` 这类入口点拆成多层文件，比如：

- ``/codemirror@6.0.1``                            -> 重导出 shim
- ``/codemirror@6.0.1/es2022/codemirror.mjs``      -> 真实代码（import 其它包）
- ``/@codemirror/state@6.4.1/es2022/state.mjs``    -> 真实状态包代码

这个脚本以入口点为种子，做一次 BFS，把整张依赖图都下载下来，然后把所有
``"/..."`` 形式的绝对路径改写成 ``"/static/vendor/..."``，最后写到本地。
之后浏览器只需要访问 ``http://<arena>/static/vendor/codemirror@6.0.1.js``
就能用整套 CodeMirror，**完全无需访问公网**。

幂等：再跑一次只会覆盖已有文件。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PKG_DIR / "static" / "vendor"
LOCAL_PREFIX = "/static/vendor"
ESM_BASE = "https://esm.sh"

# 注意：必须让所有 import 收敛到 codemirror 自己的依赖图，不能再额外 pin
# state/view/commands 的具体版本，否则会和 codemirror 内部用的 ``@^6.0.0``
# shim 指向两个不同的 state.mjs，触发"Facet 不属于这个 State"的错误。
# 我们只 pin 顶层的 codemirror / lang-python / lang-cpp 入口点；
# state / view / commands 会通过 codemirror 内部的 ``@^6.0.0`` 自动拉进来，
# 然后 app.js 也走同一个 ``@^6.0.0`` shim，保证整个 graph 是一份。
ENTRY_POINTS = (
    "/codemirror@6.0.1",
    "/@codemirror/lang-python@6.1.5",
    "/@codemirror/lang-cpp@6.0.2",
)

# 匹配 esm.sh 在自己生成代码里使用的绝对 URL：
#   "/codemirror@6.0.1"  "/@codemirror/state@6.4.1/es2022/state.mjs"
#   "/node/process.mjs"  "/v135/codemirror@6.0.1/...."
# 简单粗暴：所有 ``"/..."`` 都视为 esm.sh 内部路径
_PATH_IN_STRING = re.compile(r'(?P<q>["\'])/(?P<rest>[^"\']*)(?P=q)')


_KNOWN_EXTENSIONS = (".js", ".mjs", ".cjs", ".css", ".json", ".map", ".wasm")
# esm.sh 生成的代码里会写 ``/pkg@^6.0.0?target=es2022``、
# ``/pkg@~6``、``/pkg@6.0.0``、``/v135/...`` 这类路径。
_VERSION_RE = re.compile(r"^/(?:@[\w.-]+/)?[\w.-]+@[\^~><=]*[\w.-]+")


def _strip_query(path: str) -> str:
    """剥掉 ``?target=es2022`` 这类查询串，给本地落盘和去重用同一个 key。

    我们统一只镜像 ``target=es2022`` 这一个变体，所以同 bare path 不同 query
    会落到同一份本地文件，幂等且天然去重。
    """
    return path.split("?", 1)[0]


def _looks_like_esm_path(path: str) -> bool:
    """粗略判断这条字符串是不是 esm.sh 在 import 里指向的路径。

    避免把代码里随机的 ``"/x"`` 字面量也当成模块。esm.sh 生成的内部路径都满足：
      - 以 ``/@xxx/yyy@ver``、``/pkg@ver``、``/node/``、``/v数字/``、``/build/`` 开头
      - 或者就是 ``/path.mjs / .js``
    """
    if not path.startswith("/") or path.startswith("//"):
        return False
    bare = _strip_query(path)
    return (
        bool(_VERSION_RE.match(bare))
        or bare.startswith("/node/")
        or bare.startswith("/v")
        or bare.startswith("/build/")
        or bare.endswith(".mjs")
        or bare.endswith(".js")
    )


def _ensure_local_filename(path: str) -> str:
    """没有 JS 后缀的 esm.sh 入口（如 ``/codemirror@6.0.1``）落盘时补 ``.js``，
    免得 FastAPI 用错的 MIME 类型回送、或者下一层文件覆盖到目录上。

    注意：不能简单判断"有没有 .'"——版本号里也有点（``codemirror@6.0.1``），
    必须显式检查已知扩展名。
    """
    bare = _strip_query(path)
    last = bare.rsplit("/", 1)[-1].lower()
    if any(last.endswith(ext) for ext in _KNOWN_EXTENSIONS):
        return bare
    return bare + ".js"


def fetch(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "code_learn-arena-vendor/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def vendor() -> int:
    if VENDOR_DIR.exists():
        # 清掉旧的 vendor 目录，保证幂等
        shutil.rmtree(VENDOR_DIR)
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    queue: list[str] = list(ENTRY_POINTS)
    seen_bare: set[str] = set()
    count = 0

    while queue:
        path = queue.pop(0)
        bare = _strip_query(path)
        if bare in seen_bare:
            continue
        seen_bare.add(bare)

        url = ESM_BASE + path
        try:
            text = fetch(url)
        except urllib.error.HTTPError as exc:
            print(f"  ! 跳过 {path}: HTTP {exc.code}", file=sys.stderr)
            continue
        except urllib.error.URLError as exc:
            print(f"  ! 跳过 {path}: {exc}", file=sys.stderr)
            continue

        # 把所有 ``"/foo"`` 形式的内部路径改写成 ``"/static/vendor/foo[.js]"``
        def _rewrite(match: re.Match[str]) -> str:
            raw = "/" + match.group("rest")
            if not _looks_like_esm_path(raw):
                return match.group(0)
            local = LOCAL_PREFIX + _ensure_local_filename(raw)
            quote = match.group("q")
            return f"{quote}{local}{quote}"

        rewritten = _PATH_IN_STRING.sub(_rewrite, text)

        # 落盘：跟 esm.sh 上的相对路径一致，没后缀的入口补 .js
        save_path = _ensure_local_filename(path)
        local_file = VENDOR_DIR / save_path.lstrip("/")
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text(rewritten, encoding="utf-8")
        count += 1
        print(f"  [{count:>3}] {path} -> {local_file.relative_to(PKG_DIR)}")

        # 从原文本里发现新依赖
        for match in _PATH_IN_STRING.finditer(text):
            candidate = "/" + match.group("rest")
            if _looks_like_esm_path(candidate) and _strip_query(candidate) not in seen_bare:
                queue.append(candidate)

    print(f"\nDone. {count} 个文件镜像到 {VENDOR_DIR.relative_to(PKG_DIR)}/")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="只列出会下载的入口点，不真正写盘",
    )
    args = parser.parse_args()

    if args.list:
        for entry in ENTRY_POINTS:
            print(entry)
        return 0
    return 0 if vendor() > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
