from __future__ import annotations

from pathlib import Path
import random

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .catalog import discover_problems, get_problem
from .runner import load_code, reset_draft, run_problem, runtime_capabilities, save_draft

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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "Interview Practice Arena"},
    )


@app.get("/api/problems")
async def list_problems() -> dict[str, object]:
    return {
        "problems": [problem.to_dict() for problem in discover_problems()],
        "runtime": runtime_capabilities(),
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
    try:
        problem = get_problem(slug)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    selected_mode = mode or problem.default_mode
    if selected_mode not in problem.available_modes:
        raise HTTPException(status_code=400, detail="当前题目不支持这个模式。")

    code, is_draft = load_code(problem, selected_mode)
    return {
        "problem": problem.to_dict(),
        "mode": selected_mode,
        "code": code,
        "draft_exists": is_draft,
        "can_test": selected_mode == "exercise" and problem.test_file is not None,
        "can_run": True,
    }


@app.post("/api/save")
async def save_problem(payload: DraftPayload) -> dict[str, str]:
    try:
        problem = get_problem(payload.slug)
        target = save_draft(problem, payload.mode, payload.code)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"saved_to": str(target)}


@app.post("/api/reset")
async def reset_problem(payload: DraftPayload) -> dict[str, object]:
    try:
        problem = get_problem(payload.slug)
        code = reset_draft(problem, payload.mode)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"code": code}


@app.post("/api/run")
async def execute_problem(payload: RunPayload) -> dict[str, object]:
    try:
        problem = get_problem(payload.slug)
        result = run_problem(problem, payload.mode, payload.code, payload.action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()
