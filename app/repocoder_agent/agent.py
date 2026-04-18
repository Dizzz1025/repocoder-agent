from __future__ import annotations

import re
from dataclasses import dataclass

from .agents.patch_reviewer_agent import PatchReviewerAgent
from .agents.planner_agent import PlannerAgent
from .agents.repo_explorer_agent import RepoExplorerAgent
from .autofix import ErrorAutoFixer
from .critics.patch_critic import PatchCritic, PatchCritique
from .executor import CommandExecutor
from .hooks.manager import HookManager
from .llm_client import SupportsRepoCoderLLM, create_llm_client_from_env
from .memory.graph_builder import RepositoryGraphBuilder
from .memory.graph_store import RepositoryGraphStore
from .memory.history_store import RepositoryHistoryStore
from .models import AgentRunResponse, AgentTaskRequest, AppliedPatch, CommandResult, PatchInstruction
from .patcher import PatchApplier
from .planner import TaskPlanner
from .policies.uncertainty_gate import UncertaintyDecision, UncertaintyGate
from .repository import RepoSnapshot, RepositoryScanner
from .retrieval.hybrid_retriever import HybridRetriever
from .sandbox import DryRunResult, DryRunSandbox
from .skills.loader import SkillLoader
from .selectors.patch_selector import PatchSelectionResult, PatchSelector
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
        self.graph_builder = RepositoryGraphBuilder()
        self.graph_store = RepositoryGraphStore(task.repository_path)
        self.history_store = RepositoryHistoryStore(task.repository_path)
        self.hybrid_retriever = HybridRetriever()
        self.planner = TaskPlanner(llm_client=self.llm_client, start_dir=task.repository_path)
        self.patcher = PatchApplier(task.repository_path)
        self.executor = CommandExecutor(task.repository_path, timeout_sec=task.command_timeout_sec)
        self.auto_fixer = ErrorAutoFixer(task.repository_path)
        self.trace_writer = RunTraceWriter(task.repository_path)
        self.hook_manager = HookManager(task.repository_path)
        self.uncertainty_gate = UncertaintyGate(llm_client=self.llm_client)
        self.patch_critic = PatchCritic(llm_client=self.llm_client)
        self.patch_selector = PatchSelector(
            uncertainty_gate=self.uncertainty_gate,
            patch_critic=self.patch_critic,
        )
        self.dry_run_sandbox = DryRunSandbox(task.repository_path, timeout_sec=task.command_timeout_sec)
        self.skill_loader = SkillLoader(task.repository_path)
        self.repo_explorer_agent = RepoExplorerAgent(
            scanner=self.scanner,
            graph_store=self.graph_store,
            graph_builder=self.graph_builder,
            history_store=self.history_store,
            retriever=self.hybrid_retriever,
        )
        self.planner_agent = PlannerAgent(self.planner)
        self.patch_reviewer_agent = PatchReviewerAgent(self.patch_selector)

    def run(self) -> AgentRunResponse:
        effective_goal = self._goal_with_skill_context()
        initial_exploration = self.repo_explorer_agent.explore(
            goal=effective_goal,
            top_k_files=self.task.top_k_files,
        )
        snapshot = initial_exploration.snapshot
        graph = initial_exploration.graph
        graph_diff = initial_exploration.graph_diff
        retrieval_result = initial_exploration.retrieval_result
        relevant_files = list(retrieval_result.relevant_files)
        plan_result = self.planner_agent.create_plan(
            goal=effective_goal,
            relevant_files=relevant_files,
            commands=self.task.commands,
            patches=self.task.patches,
            auto_fix=self.task.auto_fix,
        )
        plan_steps = list(plan_result.plan_steps)

        applied_patches: list[AppliedPatch] = []
        command_results = []
        retrieval_trace = self._build_retrieval_trace(
            graph=graph,
            graph_diff=graph_diff,
            retrieval_result=retrieval_result,
        )
        selection_trace: dict[str, list[dict] | dict] = {
            "initial": [],
            "retry": [],
        }
        hooks_trace: dict[str, list[dict]] = {
            "pre_patch": [],
            "post_patch": [],
            "pre_command": [],
            "post_command": [],
            "run_stop": [],
        }
        sandbox_trace: dict[str, list[dict] | dict] = {
            "initial": [],
            "retry": [],
        }

        approved_initial_patches: list[PatchInstruction] = []
        if self.task.patches:
            for patch in self.task.patches:
                approved, blocked_result = self._approve_patch_candidate(
                    patch=patch,
                    relevant_files=relevant_files,
                    snapshot=snapshot,
                    phase="initial",
                )
                if approved:
                    approved_initial_patches.append(patch)
                elif blocked_result is not None:
                    applied_patches.append(blocked_result)
        else:
            initial_selection = self._select_initial_patch(
                snapshot=snapshot,
                relevant_files=relevant_files,
            )
            selection_trace["initial"].append(self._serialize_selection_result(initial_selection))
            approved_initial_patches.extend(self._selection_to_patches(initial_selection, applied_patches))

        if self.task.mode == "plan":
            plan_message = (
                "Plan mode completed without applying changes."
                if approved_initial_patches
                else "Plan mode completed without any approved patch candidates."
            )
            self._run_hooks(
                event="run_stop",
                hooks_trace=hooks_trace,
                contexts=[{"message": plan_message}],
            )
            response = AgentRunResponse(
                success=True,
                summary=snapshot.summary,
                relevant_files=relevant_files,
                plan_steps=plan_steps,
                applied_patches=applied_patches,
                command_results=[],
                iterations_used=0,
                message=plan_message,
                mode=self.task.mode,
                proposed_patches=approved_initial_patches,
            )
            self.trace_writer.write_run(
                request=self.task,
                summary=snapshot.summary,
                relevant_files=relevant_files,
                response=response,
                selection_trace=selection_trace,
                sandbox_trace=sandbox_trace,
                retrieval_trace=retrieval_trace,
                hooks_trace=hooks_trace,
            )
            return response

        if approved_initial_patches:
            pre_patch_results = self._run_hooks(
                event="pre_patch",
                hooks_trace=hooks_trace,
                contexts=[
                    {
                        "file_path": patch.file_path,
                        "operation": patch.operation,
                        "message": "initial patch candidate",
                    }
                    for patch in approved_initial_patches
                ],
            )
            if self._hooks_blocked(pre_patch_results):
                applied_patches.extend(self._hook_blocked_patch_results(approved_initial_patches, pre_patch_results))
                approved_initial_patches = []

        if approved_initial_patches:
            sandbox_result = self.dry_run_sandbox.validate_patches(
                patches=approved_initial_patches,
                commands=self.task.commands,
            )
            sandbox_trace["initial"].append(self._serialize_sandbox_result(sandbox_result, approved_initial_patches))
            if sandbox_result.success:
                patch_results = self.patcher.apply_many(approved_initial_patches)
                applied_patches.extend(patch_results)
                self._run_hooks(
                    event="post_patch",
                    hooks_trace=hooks_trace,
                    contexts=[
                        {
                            "file_path": patch_result.file_path,
                            "operation": patch_result.operation,
                            "message": patch_result.message,
                        }
                        for patch_result in patch_results
                    ],
                )
                for patch_result in patch_results:
                    self.history_store.record_patch_event(
                        file_path=patch_result.file_path,
                        operation=patch_result.operation,
                        success=patch_result.success,
                        message=patch_result.message,
                    )
            else:
                applied_patches.extend(self._sandbox_blocked_patch_results(approved_initial_patches, sandbox_result))
                approved_initial_patches = []

        state = _RunState()
        seen_patch_fingerprints: set[str] = set()
        for item in approved_initial_patches:
            seen_patch_fingerprints.add(self._patch_fingerprint(item))

        for iteration in range(1, self.task.max_iterations + 1):
            state.iterations_used = iteration
            pre_command_results = self._run_hooks(
                event="pre_command",
                hooks_trace=hooks_trace,
                contexts=[
                    {"command": command, "message": "about to execute command"}
                    for command in self.task.commands
                ],
            )
            if self._hooks_blocked(pre_command_results):
                state.message = "Command execution blocked by hooks."
                break
            current_results = self.executor.run_many(self.task.commands)
            self._run_hooks(
                event="post_command",
                hooks_trace=hooks_trace,
                contexts=[
                    {
                        "command": result.command,
                        "message": result.stderr or result.stdout,
                    }
                    for result in current_results
                ],
            )
            command_results.extend(current_results)

            failed = next((r for r in current_results if r.exit_code != 0), None)
            if failed is not None:
                self.history_store.record_command_failure(
                    command=failed.command,
                    stderr=failed.stderr,
                    stdout=failed.stdout,
                )
            if failed is None:
                state.success = True
                state.message = "All commands succeeded."
                break

            if not self.task.auto_fix:
                state.message = "Commands failed and auto_fix is disabled."
                break

            current_exploration = self.repo_explorer_agent.explore(
                goal=effective_goal,
                top_k_files=self.task.top_k_files,
            )
            current_snapshot = current_exploration.snapshot
            current_graph = current_exploration.graph
            current_graph_diff = current_exploration.graph_diff
            current_retrieval_result = current_exploration.retrieval_result
            current_relevant_files = list(current_retrieval_result.relevant_files)
            retrieval_trace.setdefault("retry", []).append(
                self._build_retrieval_trace(
                    graph=current_graph,
                    graph_diff=current_graph_diff,
                    retrieval_result=current_retrieval_result,
                )
            )
            retry_selection = self._select_retry_patch(
                failed=failed,
                applied_patches=applied_patches,
                current_snapshot=current_snapshot,
                current_relevant_files=current_relevant_files,
            )
            selection_trace["retry"].append(self._serialize_selection_result(retry_selection))
            approved_retry_patches = self._selection_to_patches(retry_selection, applied_patches)
            if not approved_retry_patches:
                state.message = self._selection_failure_message(
                    retry_selection,
                    default="Commands failed and no auto-fix rule matched.",
                )
                break

            pre_patch_results = self._run_hooks(
                event="pre_patch",
                hooks_trace=hooks_trace,
                contexts=[
                    {
                        "file_path": patch.file_path,
                        "operation": patch.operation,
                        "message": "retry patch candidate",
                    }
                    for patch in approved_retry_patches
                ],
            )
            if self._hooks_blocked(pre_patch_results):
                applied_patches.extend(self._hook_blocked_patch_results(approved_retry_patches, pre_patch_results))
                state.message = "Retry patch blocked by hooks."
                break

            sandbox_result = self.dry_run_sandbox.validate_patches(
                patches=approved_retry_patches,
                commands=self.task.commands,
            )
            sandbox_trace["retry"].append(self._serialize_sandbox_result(sandbox_result, approved_retry_patches))
            if not sandbox_result.success:
                applied_patches.extend(self._sandbox_blocked_patch_results(approved_retry_patches, sandbox_result))
                state.message = sandbox_result.message
                break

            suggestion = approved_retry_patches[0]
            fingerprint = self._patch_fingerprint(suggestion)
            if fingerprint in seen_patch_fingerprints:
                state.message = "Commands failed; patch selection produced a repeated patch."
                break
            seen_patch_fingerprints.add(fingerprint)

            patch_result = self.patcher.apply(suggestion)
            applied_patches.append(patch_result)
            self._run_hooks(
                event="post_patch",
                hooks_trace=hooks_trace,
                contexts=[
                    {
                        "file_path": patch_result.file_path,
                        "operation": patch_result.operation,
                        "message": patch_result.message,
                    }
                ],
            )
            self.history_store.record_patch_event(
                file_path=patch_result.file_path,
                operation=patch_result.operation,
                success=patch_result.success,
                message=patch_result.message,
            )
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
            mode=self.task.mode,
            proposed_patches=[],
        )
        self.trace_writer.write_run(
            request=self.task,
            summary=snapshot.summary,
            relevant_files=relevant_files,
            response=response,
            selection_trace=selection_trace,
            sandbox_trace=sandbox_trace,
            retrieval_trace=retrieval_trace,
            hooks_trace=hooks_trace,
        )
        return response

    def _approve_patch_candidate(
        self,
        patch: PatchInstruction,
        relevant_files,
        snapshot: RepoSnapshot,
        phase: str,
    ) -> tuple[bool, AppliedPatch | None]:
        gate_decision = self.uncertainty_gate.evaluate(
            patch=patch,
            relevant_files=relevant_files,
            phase=phase,
            goal=self.task.goal,
            snapshot=snapshot,
        )
        if not gate_decision.should_apply:
            return False, self._blocked_patch_result(
                patch=patch,
                source="uncertainty gate",
                action=gate_decision.action,
                summary=gate_decision.summary(),
            )

        critique = self.patch_critic.evaluate(
            patch=patch,
            relevant_files=relevant_files,
            snapshot=snapshot,
            phase=phase,
            goal=self.task.goal,
        )
        if not critique.should_apply:
            return False, self._critic_blocked_patch_result(patch, critique)

        return True, None

    def _select_initial_patch(
        self,
        snapshot: RepoSnapshot,
        relevant_files,
    ) -> PatchSelectionResult:
        candidates = self._collect_initial_patch_candidates(snapshot, relevant_files)
        return self.patch_reviewer_agent.review_candidates(
            candidates=candidates,
            relevant_files=relevant_files,
            snapshot=snapshot,
            goal=self.task.goal,
            phase="initial",
        ).selection

    def _select_retry_patch(
        self,
        failed: CommandResult,
        applied_patches: list[AppliedPatch],
        current_snapshot: RepoSnapshot,
        current_relevant_files,
    ) -> PatchSelectionResult:
        candidates = self._collect_retry_patch_candidates(
            failed=failed,
            applied_patches=applied_patches,
            current_snapshot=current_snapshot,
            current_relevant_files=current_relevant_files,
        )
        return self.patch_reviewer_agent.review_candidates(
            candidates=candidates,
            relevant_files=current_relevant_files,
            snapshot=current_snapshot,
            goal=self.task.goal,
            phase="retry",
        ).selection

    def _collect_initial_patch_candidates(
        self,
        snapshot: RepoSnapshot,
        relevant_files,
    ) -> list[tuple[str, PatchInstruction]]:
        candidates: list[tuple[str, PatchInstruction]] = []

        generator = getattr(self.llm_client, "generate_patch_candidates", None) if self.llm_client else None
        if callable(generator):
            for patch in generator(
                goal=self.task.goal,
                snapshot=snapshot,
                relevant_files=relevant_files,
                max_candidates=3,
            ):
                candidates.append(("llm-candidate", patch))
        elif self.llm_client is not None:
            inferred = self.llm_client.generate_patch(
                goal=self.task.goal,
                snapshot=snapshot,
                relevant_files=relevant_files,
            )
            if inferred is not None:
                candidates.append(("llm", inferred))

        rule_candidate = self._infer_patch_from_goal_rules(relevant_files)
        if rule_candidate is not None:
            candidates.append(("rule", rule_candidate))
        return candidates

    def _collect_retry_patch_candidates(
        self,
        failed: CommandResult,
        applied_patches: list[AppliedPatch],
        current_snapshot: RepoSnapshot,
        current_relevant_files,
    ) -> list[tuple[str, PatchInstruction]]:
        candidates: list[tuple[str, PatchInstruction]] = []

        generator = (
            getattr(self.llm_client, "reflect_and_suggest_fix_candidates", None)
            if self.llm_client is not None
            else None
        )
        if callable(generator):
            for patch in generator(
                goal=self.task.goal,
                command_result=failed,
                snapshot=current_snapshot,
                relevant_files=current_relevant_files,
                applied_patch_summaries=[self._summarize_applied_patch(item) for item in applied_patches],
                max_candidates=3,
            ):
                candidates.append(("llm-candidate", patch))
        elif self.llm_client is not None:
            llm_suggestion = self.llm_client.reflect_and_suggest_fix(
                goal=self.task.goal,
                command_result=failed,
                snapshot=current_snapshot,
                relevant_files=current_relevant_files,
                applied_patch_summaries=[self._summarize_applied_patch(item) for item in applied_patches],
            )
            if llm_suggestion is not None and llm_suggestion.patch is not None:
                candidates.append(("llm", llm_suggestion.patch))

        rule_patch = self.auto_fixer.suggest_fix(failed)
        if rule_patch is not None:
            candidates.append(("rule-autofix", rule_patch))
        return candidates

    def _selection_to_patches(
        self,
        selection: PatchSelectionResult,
        applied_patches: list[AppliedPatch],
    ) -> list[PatchInstruction]:
        if selection.selected_patch is not None:
            return [selection.selected_patch]

        for evaluation in selection.evaluations:
            if evaluation.critique is not None:
                applied_patches.append(self._critic_blocked_patch_result(evaluation.patch, evaluation.critique))
            else:
                applied_patches.append(
                    self._blocked_patch_result(
                        patch=evaluation.patch,
                        source="uncertainty gate",
                        action=evaluation.gate_decision.action,
                        summary=evaluation.gate_decision.summary(),
                    )
                )
        return []

    def _selection_failure_message(
        self,
        selection: PatchSelectionResult,
        default: str,
    ) -> str:
        if selection.selected_patch is not None:
            return default
        if not selection.evaluations:
            return default
        details = []
        for evaluation in selection.evaluations:
            if evaluation.critique is not None:
                details.append(f"{evaluation.patch.file_path}: patch critic -> {evaluation.critique.summary()}")
            else:
                details.append(
                    f"{evaluation.patch.file_path}: uncertainty gate -> {evaluation.gate_decision.summary()}"
                )
        return default if not details else default + " | " + " | ".join(details)

    def _goal_with_skill_context(self) -> str:
        if not self.task.skill:
            return self.task.goal
        skill = self.skill_loader.get_skill(self.task.skill)
        if skill is None:
            return self.task.goal
        return f"{self.task.goal}\n\nSkill Context ({skill.name}):\n{skill.content}"
    def _run_hooks(
        self,
        event: str,
        hooks_trace: dict[str, list[dict]],
        contexts: list[dict],
    ) -> list[dict]:
        results: list[dict] = []
        for context in contexts:
            for result in self.hook_manager.handle(event, context):
                entry = {
                    "event": result.event,
                    "action": result.action,
                    "message": result.message,
                    "matched_rule": result.matched_rule,
                    "blocked": result.blocked,
                    "context": context,
                }
                hooks_trace.setdefault(event, []).append(entry)
                results.append(entry)
        return results

    def _hooks_blocked(self, results: list[dict]) -> bool:
        return any(bool(item.get("blocked")) for item in results)

    def _hook_blocked_patch_results(
        self,
        patches: list[PatchInstruction],
        hook_results: list[dict],
    ) -> list[AppliedPatch]:
        message = "; ".join(item["message"] for item in hook_results if item.get("blocked")) or "patch blocked by hook"
        return [
            AppliedPatch(
                file_path=patch.file_path,
                operation=patch.operation,
                success=False,
                message=f"Patch blocked by hooks: {message}",
            )
            for patch in patches
        ]

    def _build_retrieval_trace(
        self,
        graph,
        graph_diff,
        retrieval_result,
    ) -> dict:
        return {
            "graph_summary": graph.summary(),
            "graph_diff": graph_diff,
            "history_summary": {
                "patch_success_counts": self.history_store.patch_success_counts(),
                "patch_failure_counts": self.history_store.patch_failure_counts(),
                "command_failure_counts": self.history_store.command_failure_counts(),
                "patch_history_event_count": len(self.history_store.patch_history_events()),
                "command_failure_event_count": len(self.history_store.command_failure_events()),
            },
            "relevant_files": [
                {
                    "file_path": item.file_path,
                    "score": item.total_score,
                    "reason": item.reason,
                    "score_breakdown": item.score_breakdown,
                }
                for item in retrieval_result.evaluations
            ],
        }

    def _serialize_selection_result(self, selection: PatchSelectionResult) -> dict:
        return {
            "selected_source": selection.selected_source,
            "selected_patch": (
                selection.selected_patch.model_dump(mode="json", exclude_none=True)
                if selection.selected_patch is not None
                else None
            ),
            "selected_score": selection.selected_score(),
            "evaluations": [
                {
                    "source": evaluation.source,
                    "patch": evaluation.patch.model_dump(mode="json", exclude_none=True),
                    "gate": {
                        "action": evaluation.gate_decision.action,
                        "reasons": list(evaluation.gate_decision.reasons),
                    },
                    "critic": (
                        {
                            "action": evaluation.critique.action,
                            "reasons": list(evaluation.critique.reasons),
                            "score": evaluation.critique.score,
                        }
                        if evaluation.critique is not None
                        else None
                    ),
                    "score": evaluation.score,
                    "score_breakdown": evaluation.score_breakdown or {},
                }
                for evaluation in selection.evaluations
            ],
        }

    def _serialize_sandbox_result(
        self,
        result: DryRunResult,
        patches: list[PatchInstruction],
    ) -> dict:
        return {
            "success": result.success,
            "message": result.message,
            "patches": [patch.model_dump(mode="json", exclude_none=True) for patch in patches],
            "patch_results": [item.model_dump(mode="json") for item in result.patch_results],
            "command_results": [item.model_dump(mode="json") for item in result.command_results],
        }

    def _infer_patch_from_goal_rules(self, relevant_files) -> PatchInstruction | None:
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

    def _blocked_patch_result(
        self,
        patch: PatchInstruction,
        source: str,
        action: str,
        summary: str,
    ) -> AppliedPatch:
        return AppliedPatch(
            file_path=patch.file_path,
            operation=patch.operation,
            success=False,
            message=f"Patch blocked by {source} ({action}): {summary}",
        )

    def _critic_blocked_patch_result(
        self,
        patch: PatchInstruction,
        critique: PatchCritique,
    ) -> AppliedPatch:
        return self._blocked_patch_result(
            patch=patch,
            source="patch critic",
            action=critique.action,
            summary=critique.summary(),
        )

    def _sandbox_blocked_patch_results(
        self,
        patches: list[PatchInstruction],
        sandbox_result: DryRunResult,
    ) -> list[AppliedPatch]:
        return [
            AppliedPatch(
                file_path=patch.file_path,
                operation=patch.operation,
                success=False,
                message=f"Patch blocked by dry-run sandbox: {sandbox_result.message}",
            )
            for patch in patches
        ]

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
