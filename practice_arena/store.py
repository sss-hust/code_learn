from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from .catalog import DATA_ROOT

STATE_FILE = DATA_ROOT / "state" / "arena_state.json"
DEFAULT_STATUS = "not_started"
STATUS_ORDER = ("not_started", "in_progress", "completed", "review")
HISTORY_LIMIT = 20


def list_progress(problem_slugs: list[str] | None = None) -> dict[str, dict[str, Any]]:
    state = load_state()
    progress = state["problems"]
    if problem_slugs is None:
        return progress
    return {slug: progress.get(slug, default_record()) for slug in problem_slugs}


def get_progress(slug: str) -> dict[str, Any]:
    state = load_state()
    return state["problems"].get(slug, default_record())


def record_open(slug: str, mode: str | None = None) -> dict[str, Any]:
    state = load_state()
    record = ensure_record(state, slug)
    record["last_opened_at"] = now_iso()
    if mode:
        record["last_mode"] = mode
    if record["status"] == DEFAULT_STATUS:
        record["status"] = "in_progress"
    save_state(state)
    return record


def update_progress(
    slug: str,
    *,
    mode: str | None = None,
    status: str | None = None,
    seconds_delta: int = 0,
) -> dict[str, Any]:
    state = load_state()
    record = ensure_record(state, slug)
    if mode:
        record["last_mode"] = mode
    if status:
        if status not in STATUS_ORDER:
            raise ValueError(f"Unsupported status: {status}")
        record["status"] = status
    if seconds_delta > 0:
        record["total_seconds"] += int(seconds_delta)
    record["updated_at"] = now_iso()
    save_state(state)
    return record


def record_solution_view(slug: str, mode: str | None = None) -> dict[str, Any]:
    state = load_state()
    record = ensure_record(state, slug)
    record["solution_revealed"] = True
    record["solution_view_count"] += 1
    record["last_solution_view_at"] = now_iso()
    if mode:
        record["last_mode"] = mode
    save_state(state)
    return record


def record_run(
    slug: str,
    *,
    mode: str,
    action: str,
    ok: bool,
    duration_ms: int,
    returncode: int,
    score: int | None = None,
) -> dict[str, Any]:
    state = load_state()
    record = ensure_record(state, slug)
    record["attempts"] += 1
    record["last_mode"] = mode
    record["last_run_at"] = now_iso()
    record["last_run_ok"] = ok
    record["last_action"] = action
    record["last_duration_ms"] = duration_ms
    record["last_returncode"] = returncode
    if score is not None:
        record["last_score"] = score
        record["best_score"] = max(record["best_score"], score)
    if ok and record["status"] == DEFAULT_STATUS:
        record["status"] = "in_progress"
    history = record.setdefault("history", [])
    history.append(
        {
            "at": now_iso(),
            "action": action,
            "ok": ok,
            "duration_ms": duration_ms,
            "returncode": returncode,
            "score": score,
        }
    )
    if len(history) > HISTORY_LIMIT:
        del history[:-HISTORY_LIMIT]
    save_state(state)
    return record


def dashboard(problem_slugs: list[str] | None = None) -> dict[str, Any]:
    progress = list_progress(problem_slugs)
    values = list(progress.values())
    return {
        "total_problems": len(progress),
        "completed": sum(1 for item in values if item["status"] == "completed"),
        "review": sum(1 for item in values if item["status"] == "review"),
        "in_progress": sum(1 for item in values if item["status"] == "in_progress"),
        "attempted": sum(1 for item in values if item["attempts"] > 0),
        "total_seconds": sum(item["total_seconds"] for item in values),
        "solution_viewed": sum(1 for item in values if item["solution_revealed"]),
        "best_score": max((item["best_score"] for item in values), default=0),
    }


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"version": 1, "updated_at": None, "problems": {}}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    temp_file = STATE_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_file.replace(STATE_FILE)


def ensure_record(state: dict[str, Any], slug: str) -> dict[str, Any]:
    problems = state.setdefault("problems", {})
    if slug not in problems:
        problems[slug] = default_record()
    return problems[slug]


def default_record() -> dict[str, Any]:
    return {
        "status": DEFAULT_STATUS,
        "total_seconds": 0,
        "attempts": 0,
        "best_score": 0,
        "last_score": None,
        "last_mode": None,
        "last_opened_at": None,
        "last_run_at": None,
        "last_run_ok": None,
        "last_action": None,
        "last_duration_ms": None,
        "last_returncode": None,
        "solution_revealed": False,
        "solution_view_count": 0,
        "last_solution_view_at": None,
        "updated_at": None,
        "history": [],
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
