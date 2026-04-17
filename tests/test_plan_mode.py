from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from repocoder_agent import cli
from repocoder_agent.agent import RepoCoderAgent
from repocoder_agent.main import app
from repocoder_agent.models import AgentTaskRequest

client = TestClient(app)


def test_agent_plan_mode_does_not_apply_patch(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text(
        'value = "bad"\nassert value == "good"\n',
        encoding="utf-8",
    )

    task = AgentTaskRequest(
        repository_path=str(tmp_path),
        goal='in check.py replace `"bad"` with `"good"`',
        commands=["python check.py"],
        auto_fix=False,
        mode="plan",
    )
    result = RepoCoderAgent(task).run()

    assert result.success is True
    assert result.mode == "plan"
    assert result.command_results == []
    assert result.proposed_patches
    assert '"bad"' in (tmp_path / "check.py").read_text(encoding="utf-8")


def test_run_endpoint_supports_plan_mode(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text(
        'value = "bad"\nassert value == "good"\n',
        encoding="utf-8",
    )

    payload = {
        "repository_path": str(tmp_path),
        "goal": 'in check.py replace `"bad"` with `"good"`',
        "commands": ["python check.py"],
        "mode": "plan",
        "auto_fix": False,
    }
    response = client.post("/run", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["mode"] == "plan"
    assert data["proposed_patches"]
    assert data["command_results"] == []


def test_cli_run_supports_plan_mode(tmp_path: Path, capsys) -> None:
    (tmp_path / "check.py").write_text(
        'value = "bad"\nassert value == "good"\n',
        encoding="utf-8",
    )

    exit_code = cli.main([
        "run",
        str(tmp_path),
        "--goal",
        'in check.py replace `"bad"` with `"good"`',
        "--command",
        "python check.py",
        "--mode",
        "plan",
        "--no-auto-fix",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["mode"] == "plan"
    assert payload["proposed_patches"]
    assert payload["command_results"] == []
