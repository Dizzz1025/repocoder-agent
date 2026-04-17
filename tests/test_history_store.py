from __future__ import annotations

from repocoder_agent.memory.history_store import RepositoryHistoryStore


def test_history_store_records_patch_and_command_failure(tmp_path) -> None:
    store = RepositoryHistoryStore(str(tmp_path))

    store.record_patch_event("a.py", "replace", True, "ok")
    store.record_patch_event("a.py", "replace", False, "failed")
    store.record_patch_event("b.py", "append", True, "ok")
    store.record_command_failure("python -m pytest -q", "boom", "")

    assert store.patch_success_counts()["a.py"] == 1
    assert store.patch_success_counts()["b.py"] == 1
    assert store.patch_failure_counts()["a.py"] == 1
    assert store.command_failure_counts()["python -m pytest -q"] == 1
