from __future__ import annotations

import json
from pathlib import Path

from repocoder_agent import cli
from repocoder_agent.config import reset_environment_cache


def test_cli_scan_outputs_json(tmp_path: Path, capsys) -> None:
    (tmp_path / "a.py").write_text("print('ok')\n", encoding="utf-8")

    exit_code = cli.main(["scan", str(tmp_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["summary"]["indexed_count"] >= 1


def test_cli_plan_outputs_plan_steps(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    reset_environment_cache(clear_loaded_values=True)
    (tmp_path / "check.py").write_text('value = "bad"\n', encoding="utf-8")

    exit_code = cli.main([
        "plan",
        str(tmp_path),
        "--goal",
        'in check.py replace `"bad"` with `"good"`',
        "--command",
        "python check.py",
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["plan_steps"][0].startswith("Clarify goal and constraints:")


def test_cli_run_outputs_agent_response(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    reset_environment_cache(clear_loaded_values=True)
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
        "--no-auto-fix",
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["success"] is True
