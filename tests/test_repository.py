from __future__ import annotations

from pathlib import Path

from repocoder_agent.repository import RepositoryScanner


def test_scan_and_retrieve(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("math utility package", encoding="utf-8")

    scanner = RepositoryScanner(str(tmp_path))
    snapshot = scanner.scan()

    assert snapshot.summary.indexed_count >= 2
    relevant = scanner.retrieve_relevant_files(snapshot, "add function in math utils", top_k=2)
    assert relevant
    assert relevant[0].file_path.endswith("math_utils.py")
