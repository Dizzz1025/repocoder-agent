from __future__ import annotations

import argparse
import json
from typing import Sequence

import uvicorn

from .agent import RepoCoderAgent
from .memory.graph_builder import RepositoryGraphBuilder
from .memory.graph_store import RepositoryGraphStore
from .memory.history_store import RepositoryHistoryStore
from .models import AgentTaskRequest, PlanRequest, ScanRequest
from .skills.loader import SkillLoader
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

    if command == "skills":
        if args.skills_command == "list":
            payload = _skills_list(args.repository_path)
        else:
            payload = _skills_show(args.repository_path, args.name)
        print(_to_json(payload))
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
                skill=args.skill,
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
                skill=args.skill,
                auto_fix=args.auto_fix,
                command_timeout_sec=args.command_timeout_sec,
                mode=args.mode,
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

    skills = subparsers.add_parser("skills", help="Inspect available skills")
    skills.add_argument("repository_path")
    skills_subparsers = skills.add_subparsers(dest="skills_command", required=True)
    skills_subparsers.add_parser("list", help="List skills")
    show = skills_subparsers.add_parser("show", help="Show a skill")
    show.add_argument("name")

    scan = subparsers.add_parser("scan", help="Scan a repository")
    scan.add_argument("repository_path")

    plan = subparsers.add_parser("plan", help="Generate a task plan")
    plan.add_argument("repository_path")
    plan.add_argument("--goal", required=True)
    plan.add_argument("--command", dest="commands", action="append")
    plan.add_argument("--top-k-files", type=int, default=5)
    plan.add_argument("--skill")
    plan.set_defaults(commands=None)

    run = subparsers.add_parser("run", help="Execute the full agent loop")
    run.add_argument("repository_path")
    run.add_argument("--goal", required=True)
    run.add_argument("--command", dest="commands", action="append")
    run.add_argument("--top-k-files", type=int, default=5)
    run.add_argument("--max-iterations", type=int, default=3)
    run.add_argument("--command-timeout-sec", type=int, default=60)
    run.add_argument("--skill")
    run.add_argument("--mode", choices=["execute", "plan"], default="execute")
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
    history_store = RepositoryHistoryStore(request.repository_path)
    relevant_files = retriever.retrieve(
        snapshot=snapshot,
        goal=request.goal,
        graph=graph,
        top_k=request.top_k_files,
        patch_success_counts=history_store.patch_success_counts(),
        patch_failure_counts=history_store.patch_failure_counts(),
        command_failure_counts=history_store.command_failure_counts(),
        patch_history_events=history_store.patch_history_events(),
        command_failure_events=history_store.command_failure_events(),
        file_memory=history_store.file_memory(),
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


def _skills_list(repository_path: str):
    loader = SkillLoader(repository_path)
    return {
        "skills": [
            {
                "name": skill.name,
                "path": skill.path,
                "title": skill.title,
                "summary": skill.summary,
                "tags": list(skill.tags),
                "resources": [
                    {
                        "path": resource.relative_path,
                        "type": resource.resource_type,
                    }
                    for resource in skill.resources
                ],
            }
            for skill in loader.list_skills()
        ]
    }



def _skills_show(repository_path: str, name: str):
    loader = SkillLoader(repository_path)
    skill = loader.get_skill(name)
    return {
        "skill": None if skill is None else {
            "name": skill.name,
            "path": skill.path,
            "title": skill.title,
            "summary": skill.summary,
            "tags": list(skill.tags),
            "resources": [
                {
                    "path": resource.relative_path,
                    "type": resource.resource_type,
                }
                for resource in skill.resources
            ],
            "content": skill.content,
        }
    }
