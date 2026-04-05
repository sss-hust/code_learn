from __future__ import annotations

from pathlib import Path
import random

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .catalog import discover_problems, get_problem
from .review import compare_with_solution, evaluate_submission, load_solution
from .runner import load_code, reset_draft, run_problem, runtime_capabilities, save_draft
from .store import dashboard, get_progress, list_progress, record_open, record_run, record_solution_view, update_progress

PACKAGE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))

app = FastAPI(title="Interview Practice Arena")
app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")


class DraftPayload(BaseModel):
    slug: str
    mode: str
    code: str


class RunPayload(DraftPayload):
    action: str


class ProgressPayload(BaseModel):
    slug: str
    mode: str | None = None
    status: str | None = None
    seconds_delta: int = 0


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "Interview Practice Arena"},
    )


@app.get("/api/problems")
async def list_problems() -> dict[str, object]:
    problems = [problem.to_dict() for problem in discover_problems()]
    slugs = [problem["slug"] for problem in problems]
    return {
        "problems": problems,
        "runtime": runtime_capabilities(),
        "progress": list_progress(slugs),
        "dashboard": dashboard(slugs),
    }


@app.get("/api/random")
async def random_problem(category: str | None = None, mode: str = "interview") -> dict[str, str]:
    candidates = [
        problem
        for problem in discover_problems()
        if mode in problem.available_modes and (category in (None, "", "all") or problem.category == category)
    ]
    if not candidates:
        raise HTTPException(status_code=404, detail="没有符合条件的题目。")
    chosen = random.choice(candidates)
    return {"slug": chosen.slug, "mode": mode}


@app.get("/api/problem/{slug:path}")
async def load_problem(slug: str, mode: str | None = None) -> dict[str, object]:
    problem = resolve_problem(slug)
    selected_mode = mode or problem.default_mode
    if selected_mode not in problem.available_modes:
        raise HTTPException(status_code=400, detail="当前题目不支持这个模式。")

    code, is_draft = load_code(problem, selected_mode)
    progress = record_open(problem.slug, selected_mode)
    return {
        "problem": problem.to_dict(),
        "mode": selected_mode,
        "code": code,
        "draft_exists": is_draft,
        "can_test": selected_mode == "exercise" and problem.test_file is not None,
        "can_run": True,
        "solution_available": problem.solution_file is not None,
        "progress": progress,
    }


@app.post("/api/save")
async def save_problem(payload: DraftPayload) -> dict[str, object]:
    problem = resolve_problem(payload.slug)
    target = save_draft(problem, payload.mode, payload.code)
    progress = update_progress(problem.slug, mode=payload.mode)
    return {"saved_to": str(target), "progress": progress}


@app.post("/api/reset")
async def reset_problem(payload: DraftPayload) -> dict[str, object]:
    problem = resolve_problem(payload.slug)
    code = reset_draft(problem, payload.mode)
    progress = update_progress(problem.slug, mode=payload.mode)
    return {"code": code, "progress": progress}


@app.post("/api/run")
async def execute_problem(payload: RunPayload) -> dict[str, object]:
    problem = resolve_problem(payload.slug)
    try:
        result = run_problem(problem, payload.mode, payload.code, payload.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    progress = record_run(
        problem.slug,
        mode=payload.mode,
        action=payload.action,
        ok=result.ok,
        duration_ms=result.duration_ms,
        returncode=result.returncode,
    )
    data = result.to_dict()
    data["progress"] = progress
    return data


@app.post("/api/progress")
async def sync_progress(payload: ProgressPayload) -> dict[str, object]:
    resolve_problem(payload.slug)
    try:
        progress = update_progress(
            payload.slug,
            mode=payload.mode,
            status=payload.status,
            seconds_delta=payload.seconds_delta,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"progress": progress, "dashboard": dashboard([problem.slug for problem in discover_problems()])}


@app.get("/api/solution/{slug:path}")
async def fetch_solution(slug: str, mode: str | None = None) -> dict[str, object]:
    problem = resolve_problem(slug)
    try:
        filename, code = load_solution(problem)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    progress = record_solution_view(problem.slug, mode=mode)
    return {
        "solution_file": filename,
        "solution_code": code,
        "progress": progress,
    }


@app.post("/api/compare")
async def compare_problem(payload: DraftPayload) -> dict[str, object]:
    problem = resolve_problem(payload.slug)
    try:
        comparison = compare_with_solution(problem, payload.code)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    progress = record_solution_view(problem.slug, mode=payload.mode)
    comparison["progress"] = progress
    return comparison


@app.post("/api/evaluate")
async def evaluate_problem(payload: DraftPayload) -> dict[str, object]:
    problem = resolve_problem(payload.slug)
    progress = get_progress(problem.slug)
    result = evaluate_submission(
        problem,
        payload.mode,
        payload.code,
        solution_viewed=progress["solution_revealed"],
    )
    progress = record_run(
        problem.slug,
        mode=payload.mode,
        action=f"evaluate:{result.action_used}",
        ok=result.run_result.ok,
        duration_ms=result.run_result.duration_ms,
        returncode=result.run_result.returncode,
        score=result.score,
    )
    data = result.to_dict()
    data["progress"] = progress
    return data


def resolve_problem(slug: str):
    try:
        return get_problem(slug)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
