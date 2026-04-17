from __future__ import annotations

from pathlib import Path

from repocoder_agent.agent import RepoCoderAgent
from repocoder_agent.config import reset_environment_cache
from repocoder_agent.llm_client import LLMRetrySuggestion, LLMUncertaintyReview
from repocoder_agent.models import AgentTaskRequest, CommandResult, PatchInstruction, RelevantFile
from repocoder_agent.policies.uncertainty_gate import UncertaintyGate
from repocoder_agent.repository import RepoSnapshot


class RiskyRetryLLMClient:
    def build_plan(self, goal, relevant_files, commands, patches, auto_fix):
        return ["LLM plan: attempt a retry patch."]

    def generate_patch(self, goal, snapshot, relevant_files):
        return None

    def reflect_and_suggest_fix(
        self,
        goal: str,
        command_result: CommandResult,
        snapshot: RepoSnapshot,
        relevant_files,
        applied_patch_summaries,
    ) -> LLMRetrySuggestion | None:
        return LLMRetrySuggestion(
            reflection="Try editing config.py to satisfy the assertion.",
            retry_prompt="Patch config.py.",
            patch=PatchInstruction(
                file_path="config.py",
                operation="replace",
                find_text='"bad"',
                replace_text='"good"',
            ),
        )

    def review_patch_uncertainty(self, goal, patch, snapshot, relevant_files, phase):
        return None


class ReviewingGateLLMClient:
    def build_plan(self, goal, relevant_files, commands, patches, auto_fix):
        return ["LLM plan: patch review required."]

    def generate_patch(self, goal, snapshot, relevant_files):
        return None

    def reflect_and_suggest_fix(self, goal, command_result, snapshot, relevant_files, applied_patch_summaries):
        return None

    def review_patch_uncertainty(self, goal, patch, snapshot, relevant_files, phase):
        return LLMUncertaintyReview(
            action="review",
            reasons=("llm review flagged semantic risk in the proposed patch",),
            confidence=0.41,
        )


def test_uncertainty_gate_allows_minimal_relevant_patch() -> None:
    gate = UncertaintyGate()
    patch = PatchInstruction(
        file_path="check.py",
        operation="replace",
        find_text='"bad"',
        replace_text='"good"',
    )
    relevant_files = [RelevantFile(file_path="check.py", score=3.0, reason="test")]

    decision = gate.evaluate(patch=patch, relevant_files=relevant_files, phase="initial")

    assert decision.action == "allow"
    assert decision.should_apply is True


def test_uncertainty_gate_reviews_high_risk_file() -> None:
    gate = UncertaintyGate()
    patch = PatchInstruction(
        file_path="config.py",
        operation="replace",
        find_text='"bad"',
        replace_text='"good"',
    )
    relevant_files = [RelevantFile(file_path="config.py", score=3.0, reason="test")]

    decision = gate.evaluate(patch=patch, relevant_files=relevant_files, phase="initial")

    assert decision.action == "review"
    assert "high-risk" in decision.summary()


def test_uncertainty_gate_merges_llm_review_for_semantic_risk() -> None:
    gate = UncertaintyGate(llm_client=ReviewingGateLLMClient())
    snapshot = RepoSnapshot(summary=None, files=[])  # type: ignore[arg-type]
    patch = PatchInstruction(
        file_path="check.py",
        operation="replace",
        find_text='"bad"',
        replace_text='"good"',
    )
    relevant_files = [RelevantFile(file_path="check.py", score=3.0, reason="test")]

    decision = gate.evaluate(
        patch=patch,
        relevant_files=relevant_files,
        phase="initial",
        goal="fix assertion",
        snapshot=snapshot,
    )

    assert decision.action == "review"
    assert "semantic risk" in decision.summary()


def test_agent_blocks_high_risk_initial_patch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    reset_environment_cache(clear_loaded_values=True)

    (tmp_path / "config.py").write_text('VALUE = "bad"\n', encoding="utf-8")
    (tmp_path / "check.py").write_text(
        'from config import VALUE\nassert VALUE == "good"\n',
        encoding="utf-8",
    )

    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal="fix config value",
        commands=["python check.py"],
        patches=[
            PatchInstruction(
                file_path="config.py",
                operation="replace",
                find_text='"bad"',
                replace_text='"good"',
            )
        ],
        auto_fix=False,
        max_iterations=1,
    )
    result = RepoCoderAgent(task).run()

    assert result.success is False
    assert any("uncertainty gate" in item.message.lower() for item in result.applied_patches)
    assert '"bad"' in (tmp_path / "config.py").read_text(encoding="utf-8")


def test_agent_blocks_low_risk_patch_when_llm_requests_review(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    reset_environment_cache(clear_loaded_values=True)

    (tmp_path / "check.py").write_text(
        'value = "bad"\nassert value == "good"\n',
        encoding="utf-8",
    )

    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal='in check.py replace `"bad"` with `"good"`',
        commands=["python check.py"],
        auto_fix=False,
        max_iterations=1,
    )
    result = RepoCoderAgent(task, llm_client=ReviewingGateLLMClient()).run()

    assert result.success is False
    assert any("semantic risk" in item.message.lower() for item in result.applied_patches)
    assert '"bad"' in (tmp_path / "check.py").read_text(encoding="utf-8")


def test_agent_blocks_high_risk_retry_patch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    reset_environment_cache(clear_loaded_values=True)

    (tmp_path / "config.py").write_text('VALUE = "bad"\n', encoding="utf-8")
    (tmp_path / "check.py").write_text(
        'from config import VALUE\nassert VALUE == "good"\n',
        encoding="utf-8",
    )

    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal="fix config value",
        commands=["python check.py"],
        auto_fix=True,
        max_iterations=2,
    )
    result = RepoCoderAgent(task, llm_client=RiskyRetryLLMClient()).run()

    assert result.success is False
    assert "uncertainty gate" in result.message.lower()
    assert any(item.file_path == "config.py" and not item.success for item in result.applied_patches)
    assert '"bad"' in (tmp_path / "config.py").read_text(encoding="utf-8")
