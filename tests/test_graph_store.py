from __future__ import annotations

from repocoder_agent.memory.graph_builder import RepositoryGraphBuilder
from repocoder_agent.memory.graph_store import RepositoryGraphStore
from repocoder_agent.repository import RepositoryScanner


def test_graph_store_can_save_and_load(tmp_path) -> None:
    (tmp_path / "module.py").write_text(
        "import math\n\ndef add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )

    snapshot = RepositoryScanner(str(tmp_path)).scan()
    builder = RepositoryGraphBuilder()
    graph = builder.build_from_snapshot(snapshot)

    store = RepositoryGraphStore(str(tmp_path))
    store.save(graph)
    loaded = store.load()

    assert loaded is not None
    assert loaded.summary()["functions"] >= 1
    assert loaded.imported_modules_by_file()["module.py"] == {"math"}


def test_graph_store_diff_detects_reuse(tmp_path) -> None:
    (tmp_path / "module.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    scanner = RepositoryScanner(str(tmp_path))
    snapshot = scanner.scan()
    store = RepositoryGraphStore(str(tmp_path))
    graph = RepositoryGraphBuilder().build_from_snapshot(snapshot)
    store.save(graph)

    diff = store.diff(snapshot)

    assert diff["reused"] is True
    assert diff["changed_files"] == []
    assert diff["removed_files"] == []


def test_graph_store_refresh_updates_only_changed_file(tmp_path) -> None:
    (tmp_path / "a.py").write_text(
        "def first():\n    return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        "def second():\n    return 2\n",
        encoding="utf-8",
    )

    scanner = RepositoryScanner(str(tmp_path))
    builder = RepositoryGraphBuilder()
    store = RepositoryGraphStore(str(tmp_path))

    first_snapshot = scanner.scan()
    first_graph = builder.build_from_snapshot(first_snapshot)
    store.save(first_graph)

    (tmp_path / "a.py").write_text(
        "def first_updated():\n    return 1\n",
        encoding="utf-8",
    )
    second_snapshot = scanner.scan()
    refreshed_graph, diff = store.refresh(second_snapshot, builder)

    symbols = refreshed_graph.symbol_names_by_file()
    assert diff["reused"] is False
    assert diff["changed_files"] == ["a.py"]
    assert "first_updated" in symbols["a.py"]
    assert "second" in symbols["b.py"]
