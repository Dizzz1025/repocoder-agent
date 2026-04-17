from __future__ import annotations

import argparse
import json
from typing import Sequence

import uvicorn

from .agent import RepoCoderAgent
from .memory.graph_builder import RepositoryGraphBuilder
from .memory.graph_store import RepositoryGraphStore
from .models import AgentTaskRequest, PlanRequest, ScanRequest
from .planner import TaskPlanner
from .repository import RepositoryScanner
from .retrieval.hybrid_retriever import HybridRetriever


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    command = getattr(args, "command", None)
    if command in {None, "serve"}:
        _serve(host=getattr(args, "host", "127.0.0.1"), port=getattr(args, "port", 8000))
        return 0

    if command == "scan":
        payload = _scan_repository(ScanRequest(repository_path=args.repository_path))
        print(_to_json(payload))
        return 0

    if command == "plan":
        payload = _plan_task(
            PlanRequest(
                repository_path=args.repository_path,
                goal=args.goal,
                commands=args.commands,
                top_k_files=args.top_k_files,
            )
        )
        print(_to_json(payload))
        return 0

    if command == "run":
        payload = RepoCoderAgent(
            AgentTaskRequest(
                repository_path=args.repository_path,
                goal=args.goal,
                commands=args.commands,
                top_k_files=args.top_k_files,
                max_iterations=args.max_iterations,
                auto_fix=args.auto_fix,
                command_timeout_sec=args.command_timeout_sec,
            )
        ).run()
        print(_to_json(payload))
        return 0

    parser.error(f"Unsupported command: {command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repocoder", description="RepoCoder-Agent CLI")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    scan = subparsers.add_parser("scan", help="Scan a repository")
    scan.add_argument("repository_path")

    plan = subparsers.add_parser("plan", help="Generate a task plan")
    plan.add_argument("repository_path")
    plan.add_argument("--goal", required=True)
    plan.add_argument("--command", dest="commands", action="append")
    plan.add_argument("--top-k-files", type=int, default=5)
    plan.set_defaults(commands=None)

    run = subparsers.add_parser("run", help="Execute the full agent loop")
    run.add_argument("repository_path")
    run.add_argument("--goal", required=True)
    run.add_argument("--command", dest="commands", action="append")
    run.add_argument("--top-k-files", type=int, default=5)
    run.add_argument("--max-iterations", type=int, default=3)
    run.add_argument("--command-timeout-sec", type=int, default=60)
    run.add_argument("--auto-fix", dest="auto_fix", action="store_true")
    run.add_argument("--no-auto-fix", dest="auto_fix", action="store_false")
    run.set_defaults(commands=None, auto_fix=True)

    return parser


def _serve(host: str, port: int) -> None:
    uvicorn.run("repocoder_agent.main:app", host=host, port=port, reload=False)


def _scan_repository(request: ScanRequest):
    scanner = RepositoryScanner(request.repository_path)
    snapshot = scanner.scan()
    return {
        "summary": snapshot.summary.model_dump(mode="json"),
        "top_files": [item.rel_path for item in snapshot.files[:20]],
    }


def _plan_task(request: PlanRequest):
    scanner = RepositoryScanner(request.repository_path)
    snapshot = scanner.scan()
    graph_builder = RepositoryGraphBuilder()
    graph_store = RepositoryGraphStore(request.repository_path)
    graph = graph_builder.build_from_snapshot(snapshot)
    graph_store.save(graph)
    retriever = HybridRetriever()
    relevant_files = retriever.retrieve(
        snapshot=snapshot,
        goal=request.goal,
        graph=graph,
        top_k=request.top_k_files,
    )
    planner = TaskPlanner(start_dir=request.repository_path)
    plan_steps = planner.build_plan(
        goal=request.goal,
        relevant_files=relevant_files,
        commands=request.commands,
        patches=request.patches,
        auto_fix=True,
    )
    return {
        "summary": snapshot.summary.model_dump(mode="json"),
        "relevant_files": [item.model_dump(mode="json") for item in relevant_files],
        "plan_steps": plan_steps,
    }


def _to_json(payload) -> str:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, indent=2)
