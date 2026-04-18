from __future__ import annotations

from dataclasses import dataclass

from ..planner import TaskPlanner


@dataclass(frozen=True)
class PlannerResult:
    plan_steps: tuple[str, ...]


class PlannerAgent:
    def __init__(self, planner: TaskPlanner):
        self.planner = planner

    def create_plan(
        self,
        goal: str,
        relevant_files,
        commands: list[str],
        patches,
        auto_fix: bool,
    ) -> PlannerResult:
        return PlannerResult(
            plan_steps=tuple(
                self.planner.build_plan(
                    goal=goal,
                    relevant_files=relevant_files,
                    commands=commands,
                    patches=patches,
                    auto_fix=auto_fix,
                )
            )
        )
