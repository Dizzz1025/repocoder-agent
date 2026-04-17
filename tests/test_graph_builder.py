from __future__ import annotations

from repocoder_agent.memory.graph_builder import RepositoryGraphBuilder
from repocoder_agent.repository import RepositoryScanner


def test_graph_builder_extracts_symbols_imports_and_calls(tmp_path) -> None:
    (tmp_path / "module.py").write_text(
        "import math\nfrom pathlib import Path\n\nclass Counter:\n    pass\n\ndef add(a, b):\n    return math.ceil(a + b)\n",
        encoding="utf-8",
    )

    snapshot = RepositoryScanner(str(tmp_path)).scan()
    graph = RepositoryGraphBuilder().build_from_snapshot(snapshot)

    symbol_map = graph.symbol_names_by_file()
    import_map = graph.imported_modules_by_file()
    call_map = graph.call_names_by_file()

    assert "module.py" in symbol_map
    assert {"Counter", "add"}.issubset(symbol_map["module.py"])
    assert "math" in import_map["module.py"]
    assert "pathlib" in import_map["module.py"]
    assert "ceil" in call_map["module.py"]
