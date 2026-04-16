from __future__ import annotations

from .models import PatchInstruction, RelevantFile


class TaskPlanner:
    def build_plan(
        self,
        goal: str,
        relevant_files: list[RelevantFile],
        commands: list[str],
        patches: list[PatchInstruction],
        auto_fix: bool,
    ) -> list[str]:
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
