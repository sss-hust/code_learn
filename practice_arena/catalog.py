from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / ".practice_arena"

CATEGORY_ORDER = ["model_layers", "triton", "cuda"]
MODE_PREFERENCE = ("exercise", "interview")
TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<index>\d+)\s*\|\s*`?(?P<name>[^`|]+)`?\s*\|\s*(?P<knowledge>[^|]+?)\s*\|\s*(?P<difficulty>[^|]+?)\s*\|$"
)


@dataclass(frozen=True)
class ProblemRecord:
    slug: str
    category: str
    title: str
    path: str
    language: str
    default_mode: str
    available_modes: tuple[str, ...]
    editable_files: dict[str, str]
    solution_file: str | None
    test_file: str | None
    knowledge_points: str | None
    difficulty: str | None
    description: str
    category_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "category": self.category,
            "title": self.title,
            "path": self.path,
            "language": self.language,
            "default_mode": self.default_mode,
            "available_modes": list(self.available_modes),
            "editable_files": self.editable_files,
            "solution_file": self.solution_file,
            "test_file": self.test_file,
            "knowledge_points": self.knowledge_points,
            "difficulty": self.difficulty,
            "description": self.description,
            "category_summary": self.category_summary,
        }

    def filename_for_mode(self, mode: str) -> str:
        if mode not in self.editable_files:
            raise KeyError(f"Unsupported mode: {mode}")
        return self.editable_files[mode]


def get_problem(slug: str) -> ProblemRecord:
    for problem in discover_problems():
        if problem.slug == slug:
            return problem
    raise KeyError(f"Unknown problem slug: {slug}")


@lru_cache(maxsize=1)
def discover_problems() -> tuple[ProblemRecord, ...]:
    problems: list[ProblemRecord] = []
    for category in CATEGORY_ORDER:
        category_dir = REPO_ROOT / category
        if not category_dir.exists():
            continue

        readme_text = read_text(category_dir / "README.md")
        category_summary = extract_category_summary(readme_text)
        table_meta = parse_problem_table(readme_text)

        for problem_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            editable_files = collect_editable_files(problem_dir)
            if not editable_files:
                continue

            modes = tuple(mode for mode in MODE_PREFERENCE if mode in editable_files)
            default_mode = modes[0]
            language = detect_language(editable_files[default_mode])
            knowledge, difficulty = lookup_table_meta(problem_dir.name, table_meta)
            description = extract_problem_description(problem_dir, editable_files, knowledge, difficulty)

            problems.append(
                ProblemRecord(
                    slug=f"{category}/{problem_dir.name}",
                    category=category,
                    title=problem_dir.name,
                    path=str(problem_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
                    language=language,
                    default_mode=default_mode,
                    available_modes=modes,
                    editable_files=editable_files,
                    solution_file=first_existing(problem_dir, ("solution.py", "solution.cu")),
                    test_file=first_existing(problem_dir, ("test.py",)),
                    knowledge_points=knowledge,
                    difficulty=difficulty,
                    description=description,
                    category_summary=category_summary,
                )
            )

    return tuple(problems)


def collect_editable_files(problem_dir: Path) -> dict[str, str]:
    editable: dict[str, str] = {}
    for candidate in ("exercise.py", "exercise.cu", "interview.py", "interview.cu"):
        if (problem_dir / candidate).exists():
            editable[candidate.split(".", maxsplit=1)[0]] = candidate
    return editable


def detect_language(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix == ".cu":
        return "cuda"
    return suffix.lstrip(".") or "text"


def lookup_table_meta(problem_name: str, table_meta: dict[str, tuple[str, str]]) -> tuple[str | None, str | None]:
    if problem_name in table_meta:
        return table_meta[problem_name]
    short_name = problem_name.split("_", maxsplit=1)[-1]
    if short_name in table_meta:
        return table_meta[short_name]
    return None, None


def parse_problem_table(readme_text: str) -> dict[str, tuple[str, str]]:
    table_meta: dict[str, tuple[str, str]] = {}
    for line in readme_text.splitlines():
        match = TABLE_ROW_RE.match(line.strip())
        if not match:
            continue
        index = int(match.group("index"))
        short_name = match.group("name").strip()
        knowledge = match.group("knowledge").strip()
        difficulty = match.group("difficulty").strip()
        table_meta[f"{index:02d}_{short_name}"] = (knowledge, difficulty)
        table_meta[short_name] = (knowledge, difficulty)
    return table_meta


def extract_category_summary(readme_text: str) -> str:
    lines: list[str] = []
    seen_heading = False
    for raw_line in readme_text.splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            seen_heading = True
            continue
        if seen_heading and line.startswith("## "):
            break
        if line:
            lines.append(line)
    return "\n".join(lines[:8]).strip()


def extract_problem_description(
    problem_dir: Path,
    editable_files: dict[str, str],
    knowledge: str | None,
    difficulty: str | None,
) -> str:
    for mode in MODE_PREFERENCE:
        filename = editable_files.get(mode)
        if not filename:
            continue
        file_path = problem_dir / filename
        snippet = extract_python_docstring(file_path) if file_path.suffix == ".py" else extract_c_like_comment(file_path)
        if snippet:
            return snippet

    lines = [f"题目：{problem_dir.name}"]
    if knowledge:
        lines.append(f"核心知识点：{knowledge}")
    if difficulty:
        lines.append(f"难度：{difficulty}")
    lines.append("当前题目没有单独的题面注释，建议结合专题 README 和代码骨架一起练。")
    return "\n".join(lines)


def extract_python_docstring(file_path: Path) -> str:
    text = read_text(file_path)
    match = re.match(r'^\s*(?P<quote>"""|\'\'\')(?P<body>.*?)(?P=quote)', text, re.DOTALL)
    return clean_block(match.group("body")) if match else ""


def extract_c_like_comment(file_path: Path) -> str:
    text = read_text(file_path)
    block_match = re.match(r"^\s*/\*(?P<body>.*?)\*/", text, re.DOTALL)
    if block_match:
        return clean_block(block_match.group("body"))

    comment_lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if comment_lines:
                break
            continue
        if stripped.startswith("//"):
            comment_lines.append(stripped[2:].strip())
            continue
        break
    return clean_block("\n".join(comment_lines))


def clean_block(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines)


def first_existing(problem_dir: Path, filenames: tuple[str, ...]) -> str | None:
    for filename in filenames:
        if (problem_dir / filename).exists():
            return filename
    return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
