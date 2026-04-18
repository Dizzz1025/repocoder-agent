from __future__ import annotations

from dataclasses import dataclass

from ..models import AppliedPatch, PatchInstruction
from ..selectors.patch_selector import PatchSelectionResult, PatchSelector


@dataclass(frozen=True)
class PatchReviewerResult:
    selection: PatchSelectionResult


class PatchReviewerAgent:
    def __init__(self, selector: PatchSelector):
        self.selector = selector

    def review_candidates(
        self,
        candidates: list[tuple[str, PatchInstruction]],
        relevant_files,
        snapshot,
        goal: str,
        phase: str,
    ) -> PatchReviewerResult:
        return PatchReviewerResult(
            selection=self.selector.select(
                candidates=candidates,
                relevant_files=relevant_files,
                snapshot=snapshot,
                goal=goal,
                phase=phase,
            )
        )
