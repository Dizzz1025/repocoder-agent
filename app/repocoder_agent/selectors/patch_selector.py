from __future__ import annotations

from dataclasses import dataclass

from ..critics.patch_critic import PatchCritic, PatchCritique
from ..models import PatchInstruction, RelevantFile
from ..policies.uncertainty_gate import UncertaintyDecision, UncertaintyGate
from ..repository import RepoFile, RepoSnapshot


@dataclass(frozen=True)
class PatchCandidateEvaluation:
    patch: PatchInstruction
    source: str
    gate_decision: UncertaintyDecision
    critique: PatchCritique | None
    score: float | None
    score_breakdown: dict[str, float] | None = None

    @property
    def is_selected_candidate(self) -> bool: # 注意：这里是有没有资格进入最终候选集
        return self.gate_decision.should_apply and self.critique is not None and self.critique.should_apply


@dataclass(frozen=True)
class PatchSelectionResult:
    selected_patch: PatchInstruction | None
    selected_source: str | None
    evaluations: tuple[PatchCandidateEvaluation, ...]

    def selected_score(self) -> float | None:
        for evaluation in self.evaluations:
            if self.selected_patch is evaluation.patch:
                return evaluation.score
        return None


class PatchSelector:
    def __init__(self, uncertainty_gate: UncertaintyGate, patch_critic: PatchCritic):
        self.uncertainty_gate = uncertainty_gate
        self.patch_critic = patch_critic

    def select(
        self,
        candidates: list[tuple[str, PatchInstruction]],
        relevant_files: list[RelevantFile],
        snapshot: RepoSnapshot,
        goal: str,
        phase: str,
    ) -> PatchSelectionResult:
        evaluations: list[PatchCandidateEvaluation] = []
        selected_patch: PatchInstruction | None = None
        selected_source: str | None = None
        best_score: float | None = None

        for source, patch in self._deduplicate(candidates):
            gate_decision = self.uncertainty_gate.evaluate(
                patch=patch,
                relevant_files=relevant_files,
                phase=phase,
                goal=goal,
                snapshot=snapshot,
            )
            if not gate_decision.should_apply:
                evaluations.append(
                    PatchCandidateEvaluation(
                        patch=patch,
                        source=source,
                        gate_decision=gate_decision,
                        critique=None,
                        score=None,
                    )
                )
                continue

            critique = self.patch_critic.evaluate(
                patch=patch,
                relevant_files=relevant_files,
                snapshot=snapshot,
                phase=phase,
                goal=goal,
            )
            if not critique.should_apply:
                evaluations.append(
                    PatchCandidateEvaluation(
                        patch=patch,
                        source=source,
                        gate_decision=gate_decision,
                        critique=critique,
                        score=None,
                    )
                )
                continue

            final_score, breakdown = self._score_candidate(
                patch=patch,
                source=source,
                relevant_files=relevant_files,
                critique=critique,
                snapshot=snapshot,
                phase=phase,
            )
            evaluations.append(
                PatchCandidateEvaluation(
                    patch=patch,
                    source=source,
                    gate_decision=gate_decision,
                    critique=critique,
                    score=final_score,
                    score_breakdown=breakdown,
                )
            )
            if best_score is None or final_score > best_score:
                best_score = final_score
                selected_patch = patch
                selected_source = source

        return PatchSelectionResult(
            selected_patch=selected_patch,
            selected_source=selected_source,
            evaluations=tuple(evaluations),
        )

    def _deduplicate(
        self,
        candidates: list[tuple[str, PatchInstruction]],
    ) -> list[tuple[str, PatchInstruction]]:
        deduped: list[tuple[str, PatchInstruction]] = []
        seen: set[str] = set()
        for source, patch in candidates:
            fingerprint = "|".join(
                [
                    patch.file_path,
                    patch.operation,
                    patch.find_text or "",
                    patch.replace_text or "",
                    patch.content or "",
                ]
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append((source, patch))
        return deduped

    def _score_candidate(
        self,
        patch: PatchInstruction,
        source: str,
        relevant_files: list[RelevantFile],
        critique: PatchCritique,
        snapshot: RepoSnapshot,
        phase: str,
    ) -> tuple[float, dict[str, float]]:
        score = critique.score if critique.score is not None else 1.0
        breakdown: dict[str, float] = {"critic_score": score}

        rank_bonus = self._relevance_rank_bonus(patch.file_path, relevant_files)
        score += rank_bonus
        if rank_bonus:
            breakdown["relevance_rank_bonus"] = rank_bonus

        operation_bonus = self._operation_bonus(patch)
        score += operation_bonus
        if operation_bonus:
            breakdown["operation_bonus"] = operation_bonus

        precision_bonus = self._precision_bonus(patch, snapshot)
        score += precision_bonus
        if precision_bonus:
            breakdown["precision_bonus"] = precision_bonus

        compactness_adjustment = self._compactness_adjustment(patch)
        score += compactness_adjustment
        if compactness_adjustment:
            breakdown["compactness_adjustment"] = compactness_adjustment

        source_bonus = self._source_bonus(source=source, phase=phase)
        score += source_bonus
        if source_bonus:
            breakdown["source_bonus"] = source_bonus

        return score, breakdown

    def _relevance_rank_bonus(
        self,
        file_path: str,
        relevant_files: list[RelevantFile],
    ) -> float:
        for rank, item in enumerate(relevant_files, start=1):
            if item.file_path != file_path:
                continue
            if rank == 1:
                return 0.35
            if rank == 2:
                return 0.25
            if rank == 3:
                return 0.15
            return 0.05
        return -0.05

    def _operation_bonus(self, patch: PatchInstruction) -> float:
        if patch.operation == "replace":
            return 0.20
        if patch.operation == "append":
            return 0.05
        return -0.10

    def _precision_bonus(self, patch: PatchInstruction, snapshot: RepoSnapshot) -> float:
        if patch.operation != "replace" or not patch.find_text:
            return 0.0

        target_file = self._target_repo_file(snapshot, patch.file_path)
        if target_file is None:
            return 0.0

        occurrences = target_file.content.count(patch.find_text)
        if occurrences == 1:
            return 0.15
        if occurrences > 1:
            return -0.10
        return -0.15

    def _compactness_adjustment(self, patch: PatchInstruction) -> float:
        if patch.operation == "replace":
            payload_size = len(patch.find_text or "") + len(patch.replace_text or "")
        else:
            payload_size = len(patch.content or "")

        if payload_size <= 200:
            return 0.10
        if payload_size <= 800:
            return 0.0
        return -0.15

    def _source_bonus(self, source: str, phase: str) -> float:
        if source == "rule-autofix" and phase == "retry":
            return 0.15
        if source == "rule":
            return 0.05
        return 0.0

    def _target_repo_file(self, snapshot: RepoSnapshot, file_path: str) -> RepoFile | None:
        for repo_file in snapshot.files:
            if repo_file.rel_path == file_path:
                return repo_file
        return None
