from __future__ import annotations

import json
from pathlib import Path

from repocoder_agent.agent import RepoCoderAgent
from repocoder_agent.config import reset_environment_cache
from repocoder_agent.models import AgentTaskRequest, PatchInstruction
from repocoder_agent.sandbox import DryRunSandbox


def test_dry_run_sandbox_accepts_passing_patch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REPOCODER_DRY_RUN_ENABLED", "true")
    reset_environment_cache(clear_loaded_values=True)

    (tmp_path / "check.py").write_text(
        'value = "bad"\nassert value == "good"\n',
        encoding="utf-8",
    )
    sandbox = DryRunSandbox(str(tmp_path), timeout_sec=30)

    result = sandbox.validate_patches(
        patches=[
            PatchInstruction(
                file_path="check.py",
                operation="replace",
                find_text='"bad"',
                replace_text='"good"',
            )
        ],
        commands=["python check.py"],
    )

    assert result.success is True
    assert result.command_results


def test_dry_run_sandbox_rejects_failing_patch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REPOCODER_DRY_RUN_ENABLED", "true")
    reset_environment_cache(clear_loaded_values=True)

    (tmp_path / "check.py").write_text(
        'value = "bad"\nassert value == "good"\n',
        encoding="utf-8",
    )
    sandbox = DryRunSandbox(str(tmp_path), timeout_sec=30)

    result = sandbox.validate_patches(
        patches=[
            PatchInstruction(
                file_path="check.py",
                operation="replace",
                find_text='"bad"',
                replace_text='"almost_good"',
            )
        ],
        commands=["python check.py"],
    )

    assert result.success is False
    assert "validation commands failed" in result.message


def test_agent_blocks_patch_when_dry_run_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REPOCODER_DRY_RUN_ENABLED", "true")
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
        goal="fix assertion",
        commands=["python check.py"],
        patches=[
            PatchInstruction(
                file_path="check.py",
                operation="replace",
                find_text='"bad"',
                replace_text='"almost_good"',
            )
        ],
        auto_fix=False,
        max_iterations=1,
    )
    result = RepoCoderAgent(task).run()

    assert result.success is False
    assert any("dry-run sandbox" in item.message.lower() for item in result.applied_patches)
    assert '"bad"' in (tmp_path / "check.py").read_text(encoding="utf-8")


def test_trace_records_sandbox_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REPOCODER_TRACE_ENABLED", "true")
    monkeypatch.setenv("REPOCODER_DRY_RUN_ENABLED", "true")
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
    result = RepoCoderAgent(task).run()

    trace_dir = tmp_path / ".repocoder" / "runs"
    trace_files = list(trace_dir.glob("*.json"))

    assert result.success is True
    assert len(trace_files) == 1
    payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert payload["sandbox_trace"]["initial"]
    sandbox_entry = payload["sandbox_trace"]["initial"][0]
    assert sandbox_entry["success"] is True
    assert sandbox_entry["patches"][0]["file_path"] == "check.py"
