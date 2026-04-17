from __future__ import annotations

from pathlib import Path

from repocoder_agent.agent import RepoCoderAgent
from repocoder_agent.config import reset_environment_cache
from repocoder_agent.llm_client import LLMPatchCritique, LLMRetrySuggestion, LLMUncertaintyReview
from repocoder_agent.models import AgentTaskRequest, CommandResult, PatchInstruction, RelevantFile, RepositorySummary
from repocoder_agent.repository import RepoFile, RepoSnapshot
from repocoder_agent.selectors.patch_selector import PatchSelector
from repocoder_agent.policies.uncertainty_gate import UncertaintyGate
from repocoder_agent.critics.patch_critic import PatchCritic


class InitialSelectionLLMClient:
    def build_plan(self, goal, relevant_files, commands, patches, auto_fix):
        return ["LLM plan: choose the best initial patch candidate."]

    def generate_patch(self, goal, snapshot, relevant_files):
        return None

    def generate_patch_candidates(self, goal, snapshot, relevant_files, max_candidates=3):
        return [
            PatchInstruction(
                file_path="config.py",
                operation="replace",
                find_text='"bad"',
                replace_text='"good"',
            )
        ]

    def reflect_and_suggest_fix(self, goal, command_result, snapshot, relevant_files, applied_patch_summaries):
        return None

    def reflect_and_suggest_fix_candidates(self, goal, command_result, snapshot, relevant_files, applied_patch_summaries, max_candidates=3):
        return []

    def review_patch_uncertainty(self, goal, patch, snapshot, relevant_files, phase):
        return None

    def critique_patch(self, goal, patch, snapshot, relevant_files, phase):
        return None


class RetrySelectionLLMClient:
    def build_plan(self, goal, relevant_files, commands, patches, auto_fix):
        return ["LLM plan: choose the best retry patch candidate."]

    def generate_patch(self, goal, snapshot, relevant_files):
        return None

    def generate_patch_candidates(self, goal, snapshot, relevant_files, max_candidates=3):
        return []

    def reflect_and_suggest_fix(self, goal, command_result, snapshot, relevant_files, applied_patch_summaries):
        return None

    def reflect_and_suggest_fix_candidates(self, goal, command_result, snapshot, relevant_files, applied_patch_summaries, max_candidates=3):
        return [
            PatchInstruction(
                file_path="config.py",
                operation="replace",
                find_text='"bad"',
                replace_text='"good"',
            ),
            PatchInstruction(
                file_path="broken.py",
                operation="replace",
                find_text="print(missing_value)\n",
                replace_text="missing_value = None\nprint(missing_value)\n",
            ),
        ]

    def review_patch_uncertainty(self, goal, patch, snapshot, relevant_files, phase):
        if patch.file_path == "config.py":
            return LLMUncertaintyReview(action="review", reasons=("config.py is high-risk",), confidence=0.2)
        return None

    def critique_patch(self, goal, patch, snapshot, relevant_files, phase):
        if patch.file_path == "broken.py":
            return LLMPatchCritique(action="apply", reasons=("patch directly fixes the NameError",), score=0.95)
        return None


def _snapshot(files: dict[str, str]) -> RepoSnapshot:
    return RepoSnapshot(
        summary=RepositorySummary(
            repo_path="/tmp/repo",
            file_count=len(files),
            indexed_count=len(files),
            skipped_count=0,
        ),
        files=[RepoFile(rel_path=path, content=content) for path, content in files.items()],
    )


def test_patch_selector_prefers_top_ranked_relevant_file() -> None:
    selector = PatchSelector(
        uncertainty_gate=UncertaintyGate(),
        patch_critic=PatchCritic(),
    )
    snapshot = _snapshot(
        {
            "best.py": 'value = "bad"\n',
            "other.py": 'value = "bad"\n',
        }
    )
    relevant_files = [
        RelevantFile(file_path="best.py", score=8.0, reason="top relevant"),
        RelevantFile(file_path="other.py", score=3.0, reason="lower relevant"),
    ]
    candidates = [
        (
            "llm-candidate",
            PatchInstruction(
                file_path="other.py",
                operation="replace",
                find_text='"bad"',
                replace_text='"good"',
            ),
        ),
        (
            "llm-candidate",
            PatchInstruction(
                file_path="best.py",
                operation="replace",
                find_text='"bad"',
                replace_text='"good"',
            ),
        ),
    ]

    selection = selector.select(
        candidates=candidates,
        relevant_files=relevant_files,
        snapshot=snapshot,
        goal="fix assertion",
        phase="initial",
    )

    assert selection.selected_patch is not None
    assert selection.selected_patch.file_path == "best.py"
    selected_eval = next(item for item in selection.evaluations if item.patch.file_path == "best.py")
    assert selected_eval.score_breakdown is not None
    assert selected_eval.score_breakdown["relevance_rank_bonus"] > 0


def test_patch_selector_penalizes_large_create_patch() -> None:
    selector = PatchSelector(
        uncertainty_gate=UncertaintyGate(),
        patch_critic=PatchCritic(),
    )
    snapshot = _snapshot(
        {
            "existing.py": 'value = "bad"\n',
        }
    )
    relevant_files = [RelevantFile(file_path="existing.py", score=4.0, reason="relevant")]
    candidates = [
        (
            "llm-candidate",
            PatchInstruction(
                file_path="existing.py",
                operation="replace",
                find_text='"bad"',
                replace_text='"good"',
            ),
        ),
        (
            "llm-candidate",
            PatchInstruction(
                file_path="new_module.py",
                operation="create",
                content="x = 1\n" * 500,
            ),
        ),
    ]

    selection = selector.select(
        candidates=candidates,
        relevant_files=relevant_files,
        snapshot=snapshot,
        goal="fix assertion",
        phase="initial",
    )

    assert selection.selected_patch is not None
    assert selection.selected_patch.file_path == "existing.py"


def test_initial_patch_selection_prefers_rule_candidate_when_llm_candidate_is_blocked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    reset_environment_cache(clear_loaded_values=True)

    (tmp_path / "check.py").write_text(
        'value = "bad"\nassert value == "good"\n',
        encoding="utf-8",
    )
    (tmp_path / "config.py").write_text('VALUE = "bad"\n', encoding="utf-8")

    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal='in check.py replace `"bad"` with `"good"`',
        commands=["python check.py"],
        auto_fix=False,
        max_iterations=1,
    )
    result = RepoCoderAgent(task, llm_client=InitialSelectionLLMClient()).run()

    assert result.success is True
    assert '"good"' in (tmp_path / "check.py").read_text(encoding="utf-8")
    assert '"bad"' in (tmp_path / "config.py").read_text(encoding="utf-8")


def test_retry_patch_selection_picks_best_candidate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    reset_environment_cache(clear_loaded_values=True)

    (tmp_path / "broken.py").write_text("print(missing_value)\n", encoding="utf-8")
    (tmp_path / "config.py").write_text('VALUE = "bad"\n', encoding="utf-8")

    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal="fix broken script",
        commands=["python broken.py"],
        auto_fix=True,
        max_iterations=2,
    )
    result = RepoCoderAgent(task, llm_client=RetrySelectionLLMClient()).run()

    assert result.success is True
    assert "missing_value = None" in (tmp_path / "broken.py").read_text(encoding="utf-8")
    assert '"bad"' in (tmp_path / "config.py").read_text(encoding="utf-8")
