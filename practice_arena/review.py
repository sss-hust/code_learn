from __future__ import annotations

from dataclasses import dataclass
import difflib
import re
from typing import Any

from .catalog import ProblemRecord, REPO_ROOT
from .runner import RunResult, run_problem

PLACEHOLDER_PATTERNS = (
    r"\bTODO\b",
    r"\bpass\b",
    r"NotImplementedError",
    r"raise\s+NotImplementedError",
)


@dataclass(frozen=True)
class EvaluationResult:
    score: int
    grade: str
    action_used: str
    heuristics: bool
    breakdown: list[dict[str, Any]]
    placeholder_hits: list[str]
    similarity: float | None
    run_result: RunResult
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "grade": self.grade,
            "action_used": self.action_used,
            "heuristics": self.heuristics,
            "breakdown": self.breakdown,
            "placeholder_hits": self.placeholder_hits,
            "similarity": self.similarity,
            "run_result": self.run_result.to_dict(),
            "summary": self.summary,
        }


def load_solution(problem: ProblemRecord) -> tuple[str, str]:
    if not problem.solution_file:
        raise ValueError("当前题目没有参考答案文件。")
    path = REPO_ROOT / problem.path / problem.solution_file
    return problem.solution_file, path.read_text(encoding="utf-8")


def compare_with_solution(problem: ProblemRecord, code: str) -> dict[str, Any]:
    solution_file, solution_code = load_solution(problem)
    diff_lines = difflib.unified_diff(
        code.splitlines(),
        solution_code.splitlines(),
        fromfile="current",
        tofile=solution_file,
        lineterm="",
    )
    similarity = difflib.SequenceMatcher(None, normalize_code(code), normalize_code(solution_code)).ratio()
    return {
        "solution_file": solution_file,
        "solution_code": solution_code,
        "diff": "\n".join(diff_lines),
        "similarity": round(similarity, 4),
    }


def evaluate_submission(
    problem: ProblemRecord,
    mode: str,
    code: str,
    *,
    solution_viewed: bool,
) -> EvaluationResult:
    action_used = "test" if mode == "exercise" and problem.test_file else "run"
    run_result = run_problem(problem, mode, code, action_used)

    placeholders = detect_placeholders(code)
    completeness_cap = 20 if action_used == "test" else 30
    completeness_score = max(0, completeness_cap - 10 * len(placeholders))

    breakdown: list[dict[str, Any]] = []
    if action_used == "test":
        correctness = 70 if run_result.ok else 0
        breakdown.append(
            {
                "label": "Correctness",
                "score": correctness,
                "max_score": 70,
                "detail": "自动测试通过得满分，否则为 0。",
            }
        )
        blind_bonus = 10 if not solution_viewed else 0
        breakdown.append(
            {
                "label": "Blind Solve",
                "score": blind_bonus,
                "max_score": 10,
                "detail": "提交前未查看参考答案可得额外分。",
            }
        )
    else:
        runnable = 60 if run_result.ok else 0
        breakdown.append(
            {
                "label": "Runnable",
                "score": runnable,
                "max_score": 60,
                "detail": "当前文件可直接运行则得分。",
            }
        )
        blind_bonus = 10 if not solution_viewed else 0
        breakdown.append(
            {
                "label": "Blind Solve",
                "score": blind_bonus,
                "max_score": 10,
                "detail": "提交前未查看参考答案可得额外分。",
            }
        )

    breakdown.append(
        {
            "label": "Completeness",
            "score": completeness_score,
            "max_score": completeness_cap,
            "detail": "根据 `TODO` / `pass` / `NotImplementedError` 等占位符做启发式扣分。",
        }
    )

    similarity: float | None = None
    if problem.solution_file:
        _, solution_code = load_solution(problem)
        similarity = round(
            difflib.SequenceMatcher(None, normalize_code(code), normalize_code(solution_code)).ratio(),
            4,
        )
        breakdown.append(
            {
                "label": "Reference Similarity",
                "score": int(similarity * 10),
                "max_score": 10,
                "detail": "仅作为结构接近度参考，不参与总分。",
            }
        )

    score = sum(item["score"] for item in breakdown if item["label"] != "Reference Similarity")
    summary = build_summary(action_used, run_result.ok, placeholders, solution_viewed)
    return EvaluationResult(
        score=score,
        grade=grade_for_score(score),
        action_used=action_used,
        heuristics=True,
        breakdown=breakdown,
        placeholder_hits=placeholders,
        similarity=similarity,
        run_result=run_result,
        summary=summary,
    )


def detect_placeholders(code: str) -> list[str]:
    hits: list[str] = []
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, code):
            hits.append(pattern)
    return hits


def normalize_code(code: str) -> str:
    lines = [line.rstrip() for line in code.splitlines()]
    cleaned = "\n".join(line for line in lines if line.strip())
    return cleaned


def grade_for_score(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "E"


def build_summary(action_used: str, ok: bool, placeholders: list[str], solution_viewed: bool) -> str:
    parts = []
    parts.append("测试通过。" if action_used == "test" and ok else "")
    parts.append("当前代码可运行。" if action_used == "run" and ok else "")
    parts.append("仍有占位符未清理。" if placeholders else "没有发现明显占位符。")
    parts.append("已查看参考答案，盲写加分失效。" if solution_viewed else "尚未查看参考答案。")
    filtered = [item for item in parts if item]
    return " ".join(filtered)
