from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from repocoder_agent.main import app

client = TestClient(app)


def test_scan_endpoint(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('ok')\n", encoding="utf-8")
    response = client.post("/scan", json={"repository_path": str(tmp_path)})
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["indexed_count"] >= 1


def test_run_endpoint_with_patch(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text(
        'value = "bad"\nassert value == "good"\n',
        encoding="utf-8",
    )

    payload = {
        "repository_path": str(tmp_path),
        "goal": "fix assertion",
        "commands": ["python check.py"],
        "patches": [
            {
                "file_path": "check.py",
                "operation": "replace",
                "find_text": '"bad"',
                "replace_text": '"good"',
            }
        ],
        "auto_fix": False,
        "max_iterations": 2,
    }
    response = client.post("/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
