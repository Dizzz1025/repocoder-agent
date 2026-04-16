from __future__ import annotations

import re
from dataclasses import dataclass

from .autofix import ErrorAutoFixer
from .executor import CommandExecutor
from .llm_client import SupportsRepoCoderLLM, create_llm_client_from_env
from .models import AgentRunResponse, AgentTaskRequest, AppliedPatch, CommandResult, PatchInstruction
from .patcher import PatchApplier
from .planner import TaskPlanner
from .repository import RepoSnapshot, RepositoryScanner
from .tracing import RunTraceWriter


@dataclass
class _RunState:
    iterations_used: int = 0
    success: bool = False
    message: str = ""


class RepoCoderAgent:
    def __init__(self, task: AgentTaskRequest, llm_client: SupportsRepoCoderLLM | None = None):
        self.task = task
        self.scanner = RepositoryScanner(task.repository_path)
        self.llm_client = llm_client if llm_client is not None else create_llm_client_from_env(
            start_dir=task.repository_path
        )
        self.planner = TaskPlanner(llm_client=self.llm_client)
        self.patcher = PatchApplier(task.repository_path)
        self.executor = CommandExecutor(task.repository_path, timeout_sec=task.command_timeout_sec)
        self.auto_fixer = ErrorAutoFixer(task.repository_path)
        self.trace_writer = RunTraceWriter(task.repository_path)

    def run(self) -> AgentRunResponse:
        snapshot = self.scanner.scan()
        relevant_files = self.scanner.retrieve_relevant_files(
            snapshot=snapshot,
            goal=self.task.goal,
            top_k=self.task.top_k_files,
        )
        plan_steps = self.planner.build_plan(
            goal=self.task.goal,
            relevant_files=relevant_files,
            commands=self.task.commands,
            patches=self.task.patches,
            auto_fix=self.task.auto_fix,
        )

        applied_patches = []
        command_results = []

        initial_patches = list(self.task.patches)
        if not initial_patches:
            inferred = self._infer_patch_from_goal(snapshot, relevant_files)
            if inferred is not None:
                initial_patches.append(inferred)

        if initial_patches:
            applied_patches.extend(self.patcher.apply_many(initial_patches))

        state = _RunState()
        seen_patch_fingerprints: set[str] = set()
        for item in initial_patches:
            seen_patch_fingerprints.add(self._patch_fingerprint(item))

        for iteration in range(1, self.task.max_iterations + 1):
            state.iterations_used = iteration
            current_results = self.executor.run_many(self.task.commands)
            command_results.extend(current_results)

            failed = next((r for r in current_results if r.exit_code != 0), None)
            if failed is None:
                state.success = True
                state.message = "All commands succeeded."
                break

            if not self.task.auto_fix:
                state.message = "Commands failed and auto_fix is disabled."
                break

            suggestion = self._suggest_retry_patch(failed, applied_patches)
            if suggestion is None:
                state.message = "Commands failed and no auto-fix rule matched."
                break

            fingerprint = self._patch_fingerprint(suggestion)
            if fingerprint in seen_patch_fingerprints:
                state.message = "Commands failed; auto-fix produced a repeated patch."
                break
            seen_patch_fingerprints.add(fingerprint)

            patch_result = self.patcher.apply(suggestion)
            applied_patches.append(patch_result)
            if not patch_result.success:
                state.message = "Commands failed; auto-fix patch could not be applied."
                break
        else:
            state.message = "Reached max iterations before all commands passed."

        if state.success and not state.message:
            state.message = "Run completed successfully."
        if not state.success and not state.message:
            state.message = "Run completed with failures."

        response = AgentRunResponse(
            success=state.success,
            summary=snapshot.summary,
            relevant_files=relevant_files,
            plan_steps=plan_steps,
            applied_patches=applied_patches,
            command_results=command_results,
            iterations_used=state.iterations_used,
            message=state.message,
        )
        self.trace_writer.write_run(
            request=self.task,
            summary=snapshot.summary,
            relevant_files=relevant_files,
            response=response,
        )
        return response

    def _suggest_retry_patch(
        self,
        failed: CommandResult,
        applied_patches: list[AppliedPatch],
    ) -> PatchInstruction | None:
        if self.llm_client is not None:
            current_snapshot = self.scanner.scan()
            current_relevant_files = self.scanner.retrieve_relevant_files(
                snapshot=current_snapshot,
                goal=self.task.goal,
                top_k=self.task.top_k_files,
            )
            llm_suggestion = self.llm_client.reflect_and_suggest_fix(
                goal=self.task.goal,
                command_result=failed,
                snapshot=current_snapshot,
                relevant_files=current_relevant_files,
                applied_patch_summaries=[self._summarize_applied_patch(item) for item in applied_patches],
            )
            if llm_suggestion is not None and llm_suggestion.patch is not None:
                return llm_suggestion.patch

        return self.auto_fixer.suggest_fix(failed)

    def _infer_patch_from_goal(
        self,
        snapshot: RepoSnapshot,
        relevant_files,
    ) -> PatchInstruction | None:
        if self.llm_client is not None:
            inferred = self.llm_client.generate_patch(
                goal=self.task.goal,
                snapshot=snapshot,
                relevant_files=relevant_files,
            )
            if inferred is not None:
                return inferred

        english = re.search(
            r"in\s+([A-Za-z0-9_./\-]+\.py)\s+replace\s+`([^`]+)`\s+with\s+`([^`]+)`",
            self.task.goal,
            flags=re.IGNORECASE,
        )
        if english:
            return PatchInstruction(
                file_path=english.group(1).replace("\\", "/"),
                operation="replace",
                find_text=english.group(2),
                replace_text=english.group(3),
            )

        chinese = re.search(
            r"在\s*([A-Za-z0-9_./\-]+\.py)\s*将`([^`]+)`替换为`([^`]+)`",
            self.task.goal,
        )
        if chinese:
            return PatchInstruction(
                file_path=chinese.group(1).replace("\\", "/"),
                operation="replace",
                find_text=chinese.group(2),
                replace_text=chinese.group(3),
            )

        generic = re.search(r"replace\s+`([^`]+)`\s+with\s+`([^`]+)`", self.task.goal, flags=re.I)
        if generic and relevant_files:
            return PatchInstruction(
                file_path=relevant_files[0].file_path,
                operation="replace",
                find_text=generic.group(1),
                replace_text=generic.group(2),
            )
        return None

    def _summarize_applied_patch(self, patch: AppliedPatch) -> str:
        return (
            f"{patch.file_path} | {patch.operation} | success={patch.success} | "
            f"message={patch.message}"
        )

    @staticmethod
    def _patch_fingerprint(patch: PatchInstruction) -> str:
        return "|".join(
            [
                patch.file_path,
                patch.operation,
                patch.find_text or "",
                patch.replace_text or "",
                patch.content or "",
            ]
        )
