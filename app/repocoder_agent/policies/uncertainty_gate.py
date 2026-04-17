from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..llm_client import LLMUncertaintyReview, SupportsRepoCoderLLM
from ..models import PatchInstruction, RelevantFile
from ..repository import RepoSnapshot

GateAction = Literal["allow", "review", "block"]

_RISKY_PATH_MARKERS = (
    "config",
    "settings",
    "auth",
    "security",
    "secret",
    ".github",
    "docker",
    "main.py",
)
_MIN_FIND_TEXT_LEN = 3
_MAX_REPLACE_TEXT_CHARS = 2_000
_MAX_CONTENT_CHARS = 1_200


@dataclass(frozen=True)
class UncertaintyDecision:
    action: GateAction
    reasons: tuple[str, ...] = ()

    @property
    def should_apply(self) -> bool:
        return self.action == "allow"

    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "no gate concerns detected"


class UncertaintyGate:
    def __init__(self, llm_client: SupportsRepoCoderLLM | None = None):
        self.llm_client = llm_client

    def evaluate(
        self,
        patch: PatchInstruction,
        relevant_files: list[RelevantFile],
        phase: str,
        goal: str | None = None,
        snapshot: RepoSnapshot | None = None,
    ) -> UncertaintyDecision:
        rule_decision = self._evaluate_rules(
            patch=patch,
            relevant_files=relevant_files,
            phase=phase,
        )
        if rule_decision.action == "block":
            return rule_decision

        if self.llm_client is None or goal is None or snapshot is None:
            return rule_decision

        reviewer = getattr(self.llm_client, "review_patch_uncertainty", None)
        if not callable(reviewer):
            return rule_decision

        llm_review = reviewer(
            goal=goal,
            patch=patch,
            snapshot=snapshot,
            relevant_files=relevant_files,
            phase=phase,
        )
        if llm_review is None:
            return rule_decision
        return self._merge_decisions(rule_decision, llm_review)

    def _evaluate_rules(
        self,
        patch: PatchInstruction,
        relevant_files: list[RelevantFile],
        phase: str,
    ) -> UncertaintyDecision:
        block_reasons: list[str] = []
        review_reasons: list[str] = []

        normalized_path = patch.file_path.lower()
        if any(marker in normalized_path for marker in _RISKY_PATH_MARKERS):
            review_reasons.append(f"target file is high-risk: {patch.file_path}")

        relevant_paths = {item.file_path for item in relevant_files}
        if relevant_paths and patch.file_path not in relevant_paths:
            review_reasons.append(
                f"target file {patch.file_path} was not retrieved as a relevant file"
            )

        if patch.operation == "replace":
            find_text = patch.find_text or ""
            replace_text = patch.replace_text or ""
            if len(find_text.strip()) < _MIN_FIND_TEXT_LEN:
                block_reasons.append("replace find_text is too short to be reliable")
            if len(find_text) > _MAX_REPLACE_TEXT_CHARS or len(replace_text) > _MAX_REPLACE_TEXT_CHARS:
                review_reasons.append("replace payload is unusually large")

        if patch.operation in {"append", "create"}:
            content = patch.content or ""
            if len(content) > _MAX_CONTENT_CHARS:
                review_reasons.append(f"{patch.operation} content is unusually large")
            if patch.operation == "create" and any(marker in normalized_path for marker in _RISKY_PATH_MARKERS):
                block_reasons.append("create on a high-risk path requires manual review")

        if phase == "retry" and patch.operation == "create":
            review_reasons.append("retry created a brand-new file instead of editing existing code")

        if block_reasons:
            return UncertaintyDecision(action="block", reasons=tuple(block_reasons + review_reasons))
        if review_reasons:
            return UncertaintyDecision(action="review", reasons=tuple(review_reasons))
        return UncertaintyDecision(action="allow")

    def _merge_decisions(
        self,
        rule_decision: UncertaintyDecision,
        llm_review: LLMUncertaintyReview,
    ) -> UncertaintyDecision:
        combined_reasons = tuple(
            reason for reason in (*rule_decision.reasons, *llm_review.reasons) if reason
        )

        if llm_review.action == "block":
            return UncertaintyDecision(action="block", reasons=combined_reasons)
        if rule_decision.action == "review" or llm_review.action == "review":
            return UncertaintyDecision(action="review", reasons=combined_reasons)
        return UncertaintyDecision(action="allow", reasons=combined_reasons)
