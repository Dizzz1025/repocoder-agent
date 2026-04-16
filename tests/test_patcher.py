from __future__ import annotations

from pathlib import Path

from repocoder_agent.models import PatchInstruction
from repocoder_agent.patcher import PatchApplier


def test_replace_patch(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")

    applier = PatchApplier(str(tmp_path))
    result = applier.apply(
        PatchInstruction(
            file_path="sample.py",
            operation="replace",
            find_text="value = 1",
            replace_text="value = 2",
        )
    )

    assert result.success
    assert "value = 2" in target.read_text(encoding="utf-8")
