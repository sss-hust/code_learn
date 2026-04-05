from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time

from .catalog import DATA_ROOT, ProblemRecord, REPO_ROOT

DRAFT_ROOT = DATA_ROOT / "drafts"
RUN_ROOT = DATA_ROOT / "runs"


@dataclass(frozen=True)
class RunResult:
    ok: bool
    action: str
    returncode: int
    duration_ms: int
    command: str
    stdout: str
    stderr: str
    workspace: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "action": self.action,
            "returncode": self.returncode,
            "duration_ms": self.duration_ms,
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "workspace": self.workspace,
        }


def runtime_capabilities() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "pytest": importlib.util.find_spec("pytest") is not None,
        "nvcc": shutil.which("nvcc") is not None,
        "repo_root": str(REPO_ROOT),
        "draft_root": str(DRAFT_ROOT),
    }


def load_code(problem: ProblemRecord, mode: str) -> tuple[str, bool]:
    draft_file = draft_path(problem, mode)
    if draft_file.exists():
        return draft_file.read_text(encoding="utf-8"), True
    source_file = source_path(problem, mode)
    return source_file.read_text(encoding="utf-8"), False


def save_draft(problem: ProblemRecord, mode: str, code: str) -> Path:
    target = draft_path(problem, mode)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code, encoding="utf-8")
    return target


def reset_draft(problem: ProblemRecord, mode: str) -> str:
    target = draft_path(problem, mode)
    if target.exists():
        target.unlink()
    return source_path(problem, mode).read_text(encoding="utf-8")


def draft_path(problem: ProblemRecord, mode: str) -> Path:
    return DRAFT_ROOT / problem.category / problem.title / problem.filename_for_mode(mode)


def source_path(problem: ProblemRecord, mode: str) -> Path:
    return REPO_ROOT / problem.path / problem.filename_for_mode(mode)


def run_problem(problem: ProblemRecord, mode: str, code: str, action: str) -> RunResult:
    filename = problem.filename_for_mode(mode)
    workspace = prepare_workspace(problem, filename, code)

    if action == "test":
        if mode != "exercise" or not problem.test_file:
            raise ValueError("当前题目没有可复用的自动测试。")
        return run_command([sys.executable, "-m", "pytest", "-q", "-s", problem.test_file], workspace, action)

    if action == "run":
        if filename.endswith(".py"):
            return run_command([sys.executable, filename], workspace, action)
        if filename.endswith(".cu"):
            return compile_and_run_cuda(filename, "__practice_run__", workspace, action)
        raise ValueError(f"不支持的文件类型: {filename}")

    raise ValueError(f"不支持的动作: {action}")


def prepare_workspace(problem: ProblemRecord, filename: str, code: str) -> Path:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    prefix = f"{problem.category}_{problem.title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
    workspace = Path(tempfile.mkdtemp(prefix=prefix, dir=RUN_ROOT))
    target_dir = workspace / problem.title
    shutil.copytree(REPO_ROOT / problem.path, target_dir, dirs_exist_ok=True)
    (target_dir / filename).write_text(code, encoding="utf-8")
    return target_dir


def compile_and_run_cuda(filename: str, executable: str, cwd: Path, action: str) -> RunResult:
    compile_result = run_command(["nvcc", "-O2", "-o", executable, filename], cwd, f"{action}:compile")
    if not compile_result.ok:
        return compile_result
    return run_command([str(cwd / executable)], cwd, action)


def run_command(command: list[str], cwd: Path, action: str) -> RunResult:
    start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        return build_result(
            completed.returncode == 0,
            action,
            completed.returncode,
            start,
            command,
            completed.stdout,
            completed.stderr,
            cwd,
        )
    except FileNotFoundError as exc:
        return build_result(False, action, 127, start, command, "", f"命令不存在: {exc}", cwd)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        message = f"{stderr}\n命令执行超时（180s）。".strip()
        return build_result(False, action, 124, start, command, stdout, message, cwd)


def build_result(
    ok: bool,
    action: str,
    returncode: int,
    start: float,
    command: list[str],
    stdout: str,
    stderr: str,
    cwd: Path,
) -> RunResult:
    return RunResult(
        ok=ok,
        action=action,
        returncode=returncode,
        duration_ms=int((time.perf_counter() - start) * 1000),
        command=" ".join(command),
        stdout=stdout,
        stderr=stderr,
        workspace=str(cwd),
    )
