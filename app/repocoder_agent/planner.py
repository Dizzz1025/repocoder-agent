from __future__ import annotations

from .llm_client import SupportsRepoCoderLLM, create_llm_client_from_env
from .models import PatchInstruction, RelevantFile


class TaskPlanner:
    def __init__(
        self,
        llm_client: SupportsRepoCoderLLM | None = None,
        start_dir: str | None = None,
    ):
        self.llm_client = llm_client if llm_client is not None else create_llm_client_from_env(
            start_dir=start_dir
        )

    def build_plan(
        self,
        goal: str,
        relevant_files: list[RelevantFile],
        commands: list[str],
        patches: list[PatchInstruction],
        auto_fix: bool,
    ) -> list[str]:
        if self.llm_client is not None:
            llm_plan = self.llm_client.build_plan(
                goal=goal,
                relevant_files=relevant_files,
                commands=commands,
                patches=patches,
                auto_fix=auto_fix,
            )
            if llm_plan:
                return llm_plan

        steps = [
            f"Clarify goal and constraints: {goal}",
            "Scan repository and index Python-centric files.",
        ]

        if relevant_files:
            files = ", ".join(item.file_path for item in relevant_files)
            steps.append(f"Inspect relevant files first: {files}")
        else:
            steps.append("No strong file matches found; rely on provided patch instructions.")

        if patches:
            steps.append(f"Apply {len(patches)} minimal patch instruction(s).")
        else:
            steps.append("Attempt to infer a minimal patch from the task description.")

        steps.append(f"Run validation commands: {', '.join(commands)}")

        if auto_fix:
            steps.append("If commands fail, generate one minimal auto-fix patch per iteration.")
        else:
            steps.append("Stop immediately if commands fail (auto-fix disabled).")

        steps.append("Return structured run report with patches and command outputs.")
        return steps
