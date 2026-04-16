from __future__ import annotations

from pathlib import Path

from repocoder_agent.agent import RepoCoderAgent
from repocoder_agent.models import AgentTaskRequest


def test_agent_can_autofix_name_error(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("print(missing_value)\n", encoding="utf-8")

    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal="fix broken script",
        commands=["python broken.py"],
        max_iterations=3,
        auto_fix=True,
    )
    result = RepoCoderAgent(task).run()

    assert result.success
    assert any(item.operation == "replace" and item.success for item in result.applied_patches)
    assert "missing_value = None" in (tmp_path / "broken.py").read_text(encoding="utf-8")
