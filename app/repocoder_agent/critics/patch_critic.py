from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..llm_client import LLMPatchCritique, SupportsRepoCoderLLM
from ..models import PatchInstruction, RelevantFile
from ..repository import RepoSnapshot

CriticAction = Literal["apply", "review", "reject"]
_MAX_PATCH_TEXT_CHARS = 4_000


@dataclass(frozen=True)
class PatchCritique:
    action: CriticAction
    reasons: tuple[str, ...] = ()
    score: float | None = None

    @property
    def should_apply(self) -> bool:
        return self.action == "apply"

    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "no critic concerns detected"


class PatchCritic:
    def __init__(self, llm_client: SupportsRepoCoderLLM | None = None):
        self.llm_client = llm_client

    def evaluate(
        self,
        patch: PatchInstruction,
        relevant_files: list[RelevantFile],
        snapshot: RepoSnapshot,
        phase: str,
        goal: str | None = None,
    ) -> PatchCritique:
        rule_critique = self._evaluate_rules(
            patch=patch,
            relevant_files=relevant_files,
            snapshot=snapshot,
            phase=phase,
        )
        if rule_critique.action == "reject":
            return rule_critique

        if self.llm_client is None or goal is None:
            return rule_critique

        reviewer = getattr(self.llm_client, "critique_patch", None)
        if not callable(reviewer):
            return rule_critique

        llm_critique = reviewer(
            goal=goal,
            patch=patch,
            snapshot=snapshot,
            relevant_files=relevant_files,
            phase=phase,
        )
        if llm_critique is None:
            return rule_critique

        return self._merge_critiques(rule_critique, llm_critique)

    def _evaluate_rules(
        self,
        patch: PatchInstruction,
        relevant_files: list[RelevantFile],
        snapshot: RepoSnapshot,
        phase: str,
    ) -> PatchCritique:
        reject_reasons: list[str] = []
        review_reasons: list[str] = []

        files_by_path = {item.rel_path: item for item in snapshot.files}
        target_file = files_by_path.get(patch.file_path)
        relevant_paths = {item.file_path for item in relevant_files}

        if patch.operation == "replace":
            if target_file is None:
                reject_reasons.append("replace target file is missing from the repository snapshot")
            find_text = patch.find_text or ""
            replace_text = patch.replace_text or ""
            if find_text == replace_text:
                reject_reasons.append("replace patch is a no-op because find_text equals replace_text")
            if len(find_text) > _MAX_PATCH_TEXT_CHARS or len(replace_text) > _MAX_PATCH_TEXT_CHARS:
                review_reasons.append("replace payload is unusually large")

            if target_file is not None:
                occurrences = target_file.content.count(find_text)
                if occurrences == 0:
                    reject_reasons.append("replace find_text does not appear in the target file")
                elif occurrences > 1:
                    review_reasons.append(
                        "replace find_text appears multiple times in the target file"
                    )

        elif patch.operation == "create":
            if target_file is not None:
                reject_reasons.append("create patch targets a file that already exists")
            if patch.content and len(patch.content) > _MAX_PATCH_TEXT_CHARS:
                review_reasons.append("create content is unusually large")

        elif patch.operation == "append":
            if patch.content and len(patch.content) > _MAX_PATCH_TEXT_CHARS:
                review_reasons.append("append content is unusually large")
            if target_file is not None and patch.content and patch.content in target_file.content:
                review_reasons.append("append content already exists in the target file")

        if relevant_paths and patch.file_path not in relevant_paths:
            review_reasons.append("patch target is outside the current relevant file set")

        if phase == "retry" and patch.operation == "create":
            review_reasons.append("retry patch creates a new file instead of updating existing code")

        if reject_reasons:
            return PatchCritique(action="reject", reasons=tuple(reject_reasons + review_reasons))
        if review_reasons:
            return PatchCritique(action="review", reasons=tuple(review_reasons))
        return PatchCritique(action="apply")

    def _merge_critiques(
        self,
        rule_critique: PatchCritique,
        llm_critique: LLMPatchCritique,
    ) -> PatchCritique:
        combined_reasons = tuple(
            reason for reason in (*rule_critique.reasons, *llm_critique.reasons) if reason
        )
        score = llm_critique.score if llm_critique.score is not None else rule_critique.score

        if llm_critique.action == "reject":
            return PatchCritique(action="reject", reasons=combined_reasons, score=score)
        if rule_critique.action == "review" or llm_critique.action == "review":
            return PatchCritique(action="review", reasons=combined_reasons, score=score)
        return PatchCritique(action="apply", reasons=combined_reasons, score=score)
