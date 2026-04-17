from __future__ import annotations

from pathlib import Path

from repocoder_agent.agent import RepoCoderAgent
from repocoder_agent.config import reset_environment_cache
from repocoder_agent.critics.patch_critic import PatchCritic
from repocoder_agent.llm_client import LLMPatchCritique
from repocoder_agent.models import AgentTaskRequest, PatchInstruction, RelevantFile, RepositorySummary
from repocoder_agent.repository import RepoSnapshot, RepoFile


class ReviewingCriticLLMClient:
    def build_plan(self, goal, relevant_files, commands, patches, auto_fix):
        return ["LLM plan: patch critique required."]

    def generate_patch(self, goal, snapshot, relevant_files):
        return None

    def reflect_and_suggest_fix(self, goal, command_result, snapshot, relevant_files, applied_patch_summaries):
        return None

    def review_patch_uncertainty(self, goal, patch, snapshot, relevant_files, phase):
        return None

    def critique_patch(self, goal, patch, snapshot, relevant_files, phase):
        return LLMPatchCritique(
            action="review",
            reasons=("llm critic found the patch semantically under-justified",),
            score=0.33,
        )


def _snapshot_with_content(file_path: str, content: str) -> RepoSnapshot:
    return RepoSnapshot(
        summary=RepositorySummary(
            repo_path="/tmp/repo",
            file_count=1,
            indexed_count=1,
            skipped_count=0,
        ),
        files=[RepoFile(rel_path=file_path, content=content)],
    )


def test_patch_critic_rejects_noop_replace() -> None:
    critic = PatchCritic()
    patch = PatchInstruction(
        file_path="check.py",
        operation="replace",
        find_text='"bad"',
        replace_text='"bad"',
    )
    snapshot = _snapshot_with_content("check.py", 'value = "bad"\n')
    relevant_files = [RelevantFile(file_path="check.py", score=2.0, reason="test")]

    critique = critic.evaluate(
        patch=patch,
        relevant_files=relevant_files,
        snapshot=snapshot,
        phase="initial",
    )

    assert critique.action == "reject"
    assert "no-op" in critique.summary()


def test_patch_critic_reviews_ambiguous_replace() -> None:
    critic = PatchCritic()
    patch = PatchInstruction(
        file_path="check.py",
        operation="replace",
        find_text='"bad"',
        replace_text='"good"',
    )
    snapshot = _snapshot_with_content('check.py', 'value = "bad"\nother = "bad"\n')
    relevant_files = [RelevantFile(file_path="check.py", score=2.0, reason="test")]

    critique = critic.evaluate(
        patch=patch,
        relevant_files=relevant_files,
        snapshot=snapshot,
        phase="initial",
    )

    assert critique.action == "review"
    assert "multiple times" in critique.summary()


def test_patch_critic_merges_llm_review() -> None:
    critic = PatchCritic(llm_client=ReviewingCriticLLMClient())
    patch = PatchInstruction(
        file_path="check.py",
        operation="replace",
        find_text='"bad"',
        replace_text='"good"',
    )
    snapshot = _snapshot_with_content('check.py', 'value = "bad"\n')
    relevant_files = [RelevantFile(file_path="check.py", score=2.0, reason="test")]

    critique = critic.evaluate(
        patch=patch,
        relevant_files=relevant_files,
        snapshot=snapshot,
        phase="initial",
        goal="fix assertion",
    )

    assert critique.action == "review"
    assert "semantically under-justified" in critique.summary()


def test_agent_blocks_patch_when_critic_requests_review(tmp_path: Path, monkeypatch) -> None:
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
    result = RepoCoderAgent(task, llm_client=ReviewingCriticLLMClient()).run()

    assert result.success is False
    assert any("patch critic" in item.message.lower() for item in result.applied_patches)
    assert '"bad"' in (tmp_path / "check.py").read_text(encoding="utf-8")
