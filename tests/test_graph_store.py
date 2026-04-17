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
