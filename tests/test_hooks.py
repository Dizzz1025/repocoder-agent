from __future__ import annotations

import json
from pathlib import Path

from repocoder_agent.agent import RepoCoderAgent
from repocoder_agent.config import reset_environment_cache
from repocoder_agent.models import AgentTaskRequest, PatchInstruction


def test_hook_blocks_pre_patch_on_config_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REPOCODER_TRACE_ENABLED", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_environment_cache(clear_loaded_values=True)

    (tmp_path / ".repocoder").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".repocoder" / "hooks.json").write_text(
        json.dumps(
            {
                "pre_patch": [
                    {
                        "name": "block-check-patch",
                        "target_file_contains": "check.py",
                        "action": "block",
                        "message": "check.py patches require manual review",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "check.py").write_text('value = "bad"\nassert value == "good"\n', encoding="utf-8")

    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal="fix check value",
        commands=["python check.py"],
        patches=[
            PatchInstruction(
                file_path="check.py",
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
    assert any("hooks" in item.message.lower() for item in result.applied_patches)
    assert '"bad"' in (tmp_path / "check.py").read_text(encoding="utf-8")


def test_trace_records_hook_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REPOCODER_TRACE_ENABLED", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_environment_cache(clear_loaded_values=True)

    (tmp_path / ".repocoder").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".repocoder" / "hooks.json").write_text(
        json.dumps(
            {
                "run_stop": [
                    {
                        "name": "log-stop",
                        "message_contains": "Plan mode",
                        "action": "log",
                        "message": "run_stop hook captured plan-mode completion",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "check.py").write_text('value = "bad"\nassert value == "good"\n', encoding="utf-8")

    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal='in check.py replace `"bad"` with `"good"`',
        commands=["python check.py"],
        auto_fix=False,
        mode="plan",
    )
    result = RepoCoderAgent(task).run()

    trace_dir = tmp_path / ".repocoder" / "runs"
    trace_files = list(trace_dir.glob("*.json"))

    assert result.success is True
    assert trace_files
    payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert payload["hooks_trace"]["run_stop"]
    assert payload["hooks_trace"]["run_stop"][0]["matched_rule"] == "log-stop"
