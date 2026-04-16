from __future__ import annotations

from pathlib import Path

from repocoder_agent.agent import RepoCoderAgent
from repocoder_agent.llm_client import LLMRetrySuggestion
from repocoder_agent.models import AgentTaskRequest, CommandResult, PatchInstruction
from repocoder_agent.repository import RepoSnapshot


class MockLLMClient:
    def __init__(self) -> None:
        self.plan_calls = 0
        self.patch_calls = 0
        self.reflect_calls = 0

    def build_plan(self, goal, relevant_files, commands, patches, auto_fix):
        self.plan_calls += 1
        return [
            "LLM plan: inspect relevant files.",
            "LLM plan: generate a minimal patch.",
            "LLM plan: rerun validation and retry if needed.",
        ]

    def generate_patch(
        self,
        goal: str,
        snapshot: RepoSnapshot,
        relevant_files,
    ) -> PatchInstruction | None:
        self.patch_calls += 1
        return PatchInstruction(
            file_path="check.py",
            operation="replace",
            find_text='"bad"',
            replace_text='"almost_good"',
        )

    def reflect_and_suggest_fix(
        self,
        goal: str,
        command_result: CommandResult,
        snapshot: RepoSnapshot,
        relevant_files,
        applied_patch_summaries,
    ) -> LLMRetrySuggestion | None:
        self.reflect_calls += 1
        check_file = next(item for item in snapshot.files if item.rel_path == "check.py")
        if '"almost_good"' not in check_file.content:
            return None
        return LLMRetrySuggestion(
            reflection="The first patch improved the value but did not satisfy the assertion.",
            retry_prompt="Replace almost_good with good.",
            patch=PatchInstruction(
                file_path="check.py",
                operation="replace",
                find_text='"almost_good"',
                replace_text='"good"',
            ),
        )


def test_agent_uses_rule_fallback_without_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

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
    result = RepoCoderAgent(task).run()

    assert result.success is True
    assert result.plan_steps[0].startswith("Clarify goal and constraints:")
    assert any(item.success for item in result.applied_patches)
    assert '"good"' in (tmp_path / "check.py").read_text(encoding="utf-8")


def test_agent_can_run_llm_plan_patch_and_retry_flow(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text(
        'value = "bad"\nassert value == "good"\n',
        encoding="utf-8",
    )

    llm_client = MockLLMClient()
    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal="make check.py pass without changing the assertion",
        commands=["python check.py"],
        auto_fix=True,
        max_iterations=3,
    )
    result = RepoCoderAgent(task, llm_client=llm_client).run()

    assert result.success is True
    assert result.plan_steps[0].startswith("LLM plan:")
    assert llm_client.plan_calls == 1
    assert llm_client.patch_calls == 1
    assert llm_client.reflect_calls >= 1
    assert len(result.applied_patches) == 2
    assert result.command_results[0].exit_code != 0
    assert result.command_results[-1].exit_code == 0
    assert '"good"' in (tmp_path / "check.py").read_text(encoding="utf-8")
