from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .agent import RepoCoderAgent
from .models import (
    AgentRunResponse,
    AgentTaskRequest,
    PlanRequest,
    PlanResponse,
    ScanRequest,
    ScanResponse,
)
from .planner import TaskPlanner
from .repository import RepositoryScanner

app = FastAPI(
    title="RepoCoder-Agent MVP",
    description="A lightweight coding agent for Python repositories.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/scan", response_model=ScanResponse)
def scan_repository(request: ScanRequest) -> ScanResponse:
    try:
        scanner = RepositoryScanner(request.repository_path)
        snapshot = scanner.scan()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    top_files = [item.rel_path for item in snapshot.files[:20]]
    return ScanResponse(summary=snapshot.summary, top_files=top_files)


@app.post("/plan", response_model=PlanResponse)
def plan_task(request: PlanRequest) -> PlanResponse:
    try:
        scanner = RepositoryScanner(request.repository_path)
        snapshot = scanner.scan()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    relevant_files = scanner.retrieve_relevant_files(
        snapshot=snapshot,
        goal=request.goal,
        top_k=request.top_k_files,
    )
    planner = TaskPlanner()
    plan_steps = planner.build_plan(
        goal=request.goal,
        relevant_files=relevant_files,
        commands=request.commands,
        patches=request.patches,
        auto_fix=True,
    )
    return PlanResponse(
        summary=snapshot.summary,
        relevant_files=relevant_files,
        plan_steps=plan_steps,
    )


@app.post("/run", response_model=AgentRunResponse)
def run_agent(request: AgentTaskRequest) -> AgentRunResponse:
    try:
        agent = RepoCoderAgent(request)
        return agent.run()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
